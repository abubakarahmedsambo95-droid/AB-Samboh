# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ambient expense-approval agent graph workflow."""

import base64
import json
import logging
import os
import google.auth
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Any

from google.adk.workflow import Workflow, node
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.apps import App
from google.genai import types

from expense_agent import config

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Google Cloud credentials if present and not already set
if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    try:
        _, project_id = google.auth.default()
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    except Exception:
        pass

if not os.environ.get("GOOGLE_CLOUD_LOCATION"):
    os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

if not os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"):
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"


class Expense(BaseModel):
    """Structured representation of the expense report."""
    amount: float = Field(description="The amount of the expense.")
    submitter: str = Field(description="The email of the submitter.")
    category: str = Field(description="The category of the expense.")
    description: str = Field(description="The description of the expense.")
    date: str = Field(description="The date of the expense (YYYY-MM-DD).")


class RiskReview(BaseModel):
    """LLM Risk Assessment structured output."""
    risk_score: int = Field(description="Risk score from 1 (low) to 10 (high)")
    risk_factors: list[str] = Field(description="List of identified risk factors, if any")
    explanation: str = Field(description="Explanation of the risk assessment")
    alert_raised: bool = Field(description="True if an alert should be raised based on high risk or anomalies")


def parse_event_payload(ctx: Context, node_input: Any) -> Expense:
    """Parses the incoming expense event payload.
    
    Accepts types.Content, str, or dict. Decodes the 'data' key if base64-encoded
    or parses it if it's plain JSON. Falls back to a flat dict if 'data' is not present.
    """
    text = ""
    payload = None

    if hasattr(node_input, "parts"):
        parts = [p.text for p in node_input.parts if p.text]
        text = "".join(parts).strip()
    elif isinstance(node_input, str):
        text = node_input.strip()
    elif isinstance(node_input, dict):
        payload = node_input
    else:
        raise ValueError(f"Unexpected input type to parse_event_payload: {type(node_input)}")

    if text:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse input text as JSON: {text}. Error: {e}")

    # Extract the data key or use the payload itself
    expense_data = None
    if isinstance(payload, dict) and "data" in payload:
        data_val = payload["data"]
        if isinstance(data_val, str):
            # Try base64 decoding first
            try:
                decoded_bytes = base64.b64decode(data_val, validate=True)
                decoded_str = decoded_bytes.decode("utf-8")
                expense_data = json.loads(decoded_str)
            except Exception:
                # If b64 decoding or json parsing fails, treat it as plain JSON string
                try:
                    expense_data = json.loads(data_val)
                except Exception:
                    raise ValueError(f"Could not parse 'data' string as base64 or JSON: {data_val}")
        elif isinstance(data_val, dict):
            expense_data = data_val
        else:
            raise ValueError(f"Unexpected type for 'data' key: {type(data_val)}")
    else:
        # If no "data" key, assume the payload is the expense itself
        expense_data = payload

    if not isinstance(expense_data, dict):
        raise ValueError(f"Parsed expense data is not a dictionary: {expense_data}")

    # Validate against Expense model
    try:
        expense = Expense(
            amount=float(expense_data["amount"]),
            submitter=str(expense_data["submitter"]),
            category=str(expense_data["category"]),
            description=str(expense_data["description"]),
            date=str(expense_data["date"])
        )
        return expense
    except KeyError as e:
        raise ValueError(f"Missing required expense field: {e} in {expense_data}")
    except ValueError as e:
        raise ValueError(f"Invalid expense field value: {e} in {expense_data}")


def route_expense(ctx: Context, node_input: Expense):
    """Routes the expense in Python based on the config.THRESHOLD value."""
    # Store expense dict in context state for subsequent nodes (e.g. human review)
    ctx.state["expense"] = node_input.model_dump()
    
    amount = node_input.amount
    if amount < config.THRESHOLD:
        # Under threshold -> auto-approve instantly, no LLM involved
        decision = {
            "approved": True,
            "reason": f"Auto-approved: Expense of ${amount:.2f} is under the ${config.THRESHOLD:.2f} threshold.",
            "risk_reviewed": False,
        }
        return Event(output=decision, route="auto_approved")
    else:
        # Equal or over threshold -> route to LLM risk review
        return Event(output=node_input, route="needs_review")


# The LLM risk review agent
llm_risk_review = LlmAgent(
    name="llm_risk_review",
    model=config.MODEL,
    instruction=(
        "You are a risk-compliance assistant. Review the following expense details for risk factors. "
        "Analyze the submitter, category, description, and amount. "
        "Identify specific risk factors (e.g. split transactions, personal/non-business items, "
        "suspicious descriptions, mismatches between category and description, compliance issues). "
        "Determine the risk score (1-10) and raise an alert (alert_raised = True if risk score >= 5, "
        "or if there are significant anomalies)."
    ),
    output_schema=RiskReview,
)


@node(rerun_on_resume=True)
async def human_approval(ctx: Context, node_input: dict):
    """Pauses the workflow for human approval/rejection of high-value expenses.
    
    Uses RequestInput to yield an interrupt. When resumed, reads the response.
    """
    expense_dict = ctx.state.get("expense", {})
    
    # Extract risk review results (passed from llm_risk_review)
    risk_score = node_input.get("risk_score", 0)
    risk_factors = node_input.get("risk_factors", [])
    explanation = node_input.get("explanation", "")
    alert_raised = node_input.get("alert_raised", False)

    alert_header = "🚨 ALERT: High Risk Detected! 🚨" if alert_raised else "⚠️ Review Required"
    factors_str = ", ".join(risk_factors) if risk_factors else "None"

    # Prepare message for the human reviewer
    message = (
        f"{alert_header}\n"
        f"An expense of ${expense_dict.get('amount'):.2f} submitted by {expense_dict.get('submitter')} requires review.\n\n"
        f"Expense Details:\n"
        f"  - Category: {expense_dict.get('category')}\n"
        f"  - Description: {expense_dict.get('description')}\n"
        f"  - Date: {expense_dict.get('date')}\n\n"
        f"LLM Risk Assessment:\n"
        f"  - Risk Score: {risk_score}/10 (Alert: {alert_raised})\n"
        f"  - Risk Factors: {factors_str}\n"
        f"  - Explanation: {explanation}\n\n"
        f"Do you approve this expense? (yes/no)"
    )

    if not ctx.resume_inputs or "human_decision" not in ctx.resume_inputs:
        yield RequestInput(
            interrupt_id="human_decision",
            message=message,
        )
        return

    # Process resume response
    resume_data = ctx.resume_inputs["human_decision"]
    if isinstance(resume_data, dict):
        reply = str(resume_data.get("approved", "")).strip().lower()
    else:
        reply = str(resume_data).strip().lower()

    approved = reply in ("yes", "y", "true", "approve", "approved")

    decision = {
        "approved": approved,
        "reason": f"Human review decision: {'Approved' if approved else 'Rejected'}.",
        "risk_reviewed": True,
        "risk_score": risk_score,
    }
    yield Event(output=decision)


def finalize_approval(ctx: Context, node_input: dict):
    """Finalizes the approval process and logs/emits the outcome."""
    expense = ctx.state.get("expense", {})
    approved = node_input.get("approved", False)
    reason = node_input.get("reason", "")
    risk_reviewed = node_input.get("risk_reviewed", False)
    risk_score = node_input.get("risk_score")

    status = "APPROVED" if approved else "REJECTED"
    msg = f"Expense of ${expense.get('amount'):.2f} submitted by {expense.get('submitter')} was {status}. Reason: {reason}"

    # Emit content for Web UI rendering
    yield Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)])
    )
    # Return the final structured output
    yield Event(
        output={
            "status": status,
            "expense": expense,
            "reason": reason,
            "risk_reviewed": risk_reviewed,
            "risk_score": risk_score,
        }
    )


# Wire the workflow graph
root_agent = Workflow(
    name="expense_approval_workflow",
    edges=[
        ("START", parse_event_payload),
        (parse_event_payload, route_expense),
        (route_expense, finalize_approval, "auto_approved"),
        (route_expense, llm_risk_review, "needs_review"),
        (llm_risk_review, human_approval),
        (human_approval, finalize_approval),
    ]
)

# Initialize ADK App
app = App(
    root_agent=root_agent,
    name="ambient_expense_agent",
)
