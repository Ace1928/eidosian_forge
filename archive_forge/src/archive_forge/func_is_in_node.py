from __future__ import annotations
import io
import json
from email.parser import Parser
from importlib.resources import files
from typing import TYPE_CHECKING, Any
import js  # type: ignore[import-not-found]
from pyodide.ffi import (  # type: ignore[import-not-found]
from .request import EmscriptenRequest
from .response import EmscriptenResponse
def is_in_node() -> bool:
    return hasattr(js, 'process') and hasattr(js.process, 'release') and hasattr(js.process.release, 'name') and (js.process.release.name == 'node')