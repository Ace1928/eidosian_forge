import os
import yaml
import typer
import contextlib
import subprocess
import concurrent.futures
from pathlib import Path
from pydantic import model_validator
from lazyops.types.models import BaseModel
from lazyops.libs.proxyobj import ProxyObject
from typing import Optional, List, Any, Dict, Union
def parse_text_file(path: Path) -> List[str]:
    """
    Parses a text file
    """
    text_lines = path.read_text().split('\n')
    return [line.strip() for line in text_lines if '#' not in line[:5] and line.strip()]