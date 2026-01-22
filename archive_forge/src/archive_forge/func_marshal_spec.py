from __future__ import annotations
import json
from typing import Optional, Type
import requests
import yaml
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
def marshal_spec(txt: str) -> dict:
    """Convert the yaml or json serialized spec to a dict.

    Args:
        txt: The yaml or json serialized spec.

    Returns:
        dict: The spec as a dict.
    """
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return yaml.safe_load(txt)