from __future__ import annotations
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
import httpx
import semantic_version
from huggingface_hub import HfApi
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from tomlkit import parse
from typer import Argument, Option
from typing_extensions import Annotated
def make_dockerfile(demo):
    return f'\nFROM python:3.9\n\nWORKDIR /code\n\nCOPY --link --chown=1000 . .\n\nRUN mkdir -p /tmp/cache/\nRUN chmod a+rwx -R /tmp/cache/\nENV TRANSFORMERS_CACHE=/tmp/cache/\n\nRUN pip install --no-cache-dir -r requirements.txt\n\nENV PYTHONUNBUFFERED=1 \tGRADIO_ALLOW_FLAGGING=never \tGRADIO_NUM_PORTS=1 \tGRADIO_SERVER_NAME=0.0.0.0     GRADIO_SERVER_PORT=7860 \tSYSTEM=spaces\n\nCMD ["python", "{demo}"]\n'