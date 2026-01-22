from __future__ import annotations
import asyncio
import functools
import hashlib
import io
import json
import os
import pathlib
import sys
import uuid
from typing import (
import bokeh
import js
import param
import pyodide # isort: split
from bokeh import __version__
from bokeh.core.serialization import Buffer, Serialized, Serializer
from bokeh.document import Document
from bokeh.document.json import PatchJson
from bokeh.embed.elements import script_for_render_items
from bokeh.embed.util import standalone_docs_json_and_render_items
from bokeh.embed.wrappers import wrap_in_script_tag
from bokeh.events import DocumentReady
from bokeh.io.doc import set_curdoc
from bokeh.model import Model
from bokeh.settings import settings as bk_settings
from bokeh.util.sampledata import (
from js import JSON, XMLHttpRequest
from ..config import config
from ..util import edit_readonly, isurl
from . import resources
from .document import MockSessionContext
from .loading import LOADING_INDICATOR_CSS_CLASS
from .mime_render import WriteCallbackStream, exec_with_return, format_mime
from .state import state

    Executes Python code and returns a MIME representation of the
    return value.

    Arguments
    ---------
    code: str
        Python code to execute
    stdout_callback: Callable[[str, str], None] | None
        Callback executed with output written to stdout.
    stderr_callback: Callable[[str, str], None] | None
        Callback executed with output written to stderr.
    target: str
        The ID of the DOM node to write the output into.

    Returns
    -------
    Returns an JS Map containing the content, mime_type, stdout and stderr.
    