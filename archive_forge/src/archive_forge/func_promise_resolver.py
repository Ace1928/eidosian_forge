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
def promise_resolver(js_resolve_fn: JsProxy, js_reject_fn: JsProxy) -> None:

    def onMsg(e: JsProxy) -> None:
        self.streaming_ready = True
        js_resolve_fn(e)

    def onErr(e: JsProxy) -> None:
        js_reject_fn(e)
    self.js_worker.onmessage = onMsg
    self.js_worker.onerror = onErr