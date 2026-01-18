
#!/usr/bin/env bash
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
echo "Virtual environment created. Activate with 'source .venv/bin/activate'"
