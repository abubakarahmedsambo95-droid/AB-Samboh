# ruff: noqa
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

import re
import os
import google.auth
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google.adk.workflow import Workflow
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.events.event import Event, EventActions
from google.adk.events.request_input import RequestInput
from google.adk.apps import App
from google.genai import types

# Load environment variables from .env file
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
    amount: float = Field(description="The amount of the expense.")
    merchant: str = Field(description="The merchant or store name.")
    category: str = Field(
        description="The category of the expense (e.g. travel, meals, office)."
    )


# Node 1: Parse the expense from user text input
parse_expense = LlmAgent(
    name="parse_expense",
    model="gemini-2.0-flash",
    instruction="Extract expense details (amount, merchant, category) from the text.",
    output_schema=Expense,
)


# Node 2: Check whether the expense needs manual approval
def security_check(ctx: Context, node_input: str):
    """Scrub personal data and detect prompt injection.

    Returns an Event with route 'clean' for safe descriptions or
    'security_review' if injection patterns are found.
    """
    # Patterns for SSN and credit card numbers
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    cc_pattern = r"\b(?:\d[ -]*?){13,16}\b"
    redacted_categories = []
    cleaned = node_input
    if re.search(ssn_pattern, cleaned):
        cleaned = re.sub(ssn_pattern, "[REDACTED_SSN]", cleaned)
        redacted_categories.append("SSN")
    if re.search(cc_pattern, cleaned):
        cleaned = re.sub(cc_pattern, "[REDACTED_CC]", cleaned)
        redacted_categories.append("CreditCard")
    # Simple injection detection (keywords that suggest bypass)
    injection_keywords = ["auto_approve", "ignore", "bypass", "force_approval"]
    if any(kw in cleaned.lower() for kw in injection_keywords):
        # Flag as security event and route to human
        ctx.state["redacted"] = redacted_categories
        return Event(output={"description": cleaned, "redacted": redacted_categories},
                     actions=EventActions(route="security_review"))
    # No injection detected; proceed cleanly
    ctx.state["redacted"] = redacted_categories
    return Event(output={"description": cleaned, "redacted": redacted_categories},
                 actions=EventActions(route="clean"))

# Node 2: Check whether the expense needs manual approval
def check_approval(ctx: Context, node_input: dict):

    ctx.state["expense"] = node_input

    amount = node_input.get("amount", 0)

    if amount > 100:
        return Event(
            output={"needs_approval": True, "expense": node_input},
            actions=EventActions(route="requires_approval"),
        )
    
    return Event(
        output={
            "approved": True,
            "expense": node_input,
            "reason": "Auto approved"
        },
        actions=EventActions(route="auto_approved")
    )


# Node 3: Human-in-the-Loop step using RequestInput to ask for manager approval
async def request_approval(ctx: Context, node_input: dict):
    if not node_input.get("needs_approval", False):
        return

    if not ctx.resume_inputs or "manager_approval" not in ctx.resume_inputs:
        yield RequestInput(
            interrupt_id="manager_approval",
            message=f"Expense of ${node_input.get('amount')} at {node_input.get('merchant')} requires approval. Approve? (yes/no)"
        )
        return

    # Process response from manager
    resume_data = ctx.resume_inputs.get("manager_approval", "")

    if isinstance(resume_data, dict):
        manager_reply = str(resume_data.get("approved", "")).lower()
    else:
        manager_reply = str(resume_data).lower()

    approved = manager_reply in ["yes", "true", "approve", "y"]

    expense = node_input.get("expense", ctx.state.get("expense", {}))

    decision = {
        "approved": approved,
        "reason": f"Manager decision: {'Approved' if approved else 'Rejected'}",
        "expense": expense,
    }

    yield Event(output=decision)


# Node 4: Finalize the expense and produce formatted output for user/system
def finalize(ctx: Context, node_input: dict):
    expense = ctx.state.get("expense", {})
    approved = node_input.get("approved", False)
    reason = node_input.get("reason", "")

    status = "APPROVED" if approved else "REJECTED"
    msg = f"Expense of ${expense.get('amount', 0.0)} at {expense.get('merchant', 'unknown')} was {status}. Reason: {reason}"

    # Emit content for Web UI rendering
    yield Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)])
    )
    yield Event(output={"status": status, "expense": expense, "reason": reason})


# Construct the Graph Workflow
root_agent = Workflow(
    name="ambient_expense_agent",
    edges=[
    ("START", security_check),
    (security_check, parse_expense),
    (parse_expense, check_approval),
    (check_approval, request_approval),
    (request_approval, finalize),
]
)

app = App(
    root_agent=root_agent,
    name="ambient_expense_agent",
)
