from __future__ import annotations
import ast
import base64
import copy
import io
import pathlib
import pkgutil
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from html import escape
from textwrap import dedent
from typing import Any, Dict, List
import markdown
def render_plotly(value, meta, mime):
    from ..pane import Plotly
    config = value.pop('config', {})
    return Plotly(value, config=config)