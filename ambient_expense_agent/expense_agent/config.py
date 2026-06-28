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

"""Configuration for the ambient expense approval workflow."""

# The dollar threshold for auto-approval.
# Expenses under this amount are auto-approved instantly in python code.
THRESHOLD = 100.0

# The Gemini model to use for risk assessment on expenses >= THRESHOLD.
MODEL = "gemini-3.1-flash-lite"
