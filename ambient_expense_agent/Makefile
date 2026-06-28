## Makefile for Ambient Expense Agent

.PHONY: install playground

# Install project dependencies using uv (ensures reproducible environment)
install:
	@echo "Installing dependencies..."
	C:\Users\HomePC\.local\bin\uv.exe sync --all-extras

# Launch the ADK Playground UI in the background
playground:
	@echo "Starting ADK Playground..."
	C:\Users\HomePC\.local\bin\agents-cli.exe playground
