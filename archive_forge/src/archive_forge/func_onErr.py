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
def onErr(e: JsProxy) -> None:
    js_reject_fn(e)