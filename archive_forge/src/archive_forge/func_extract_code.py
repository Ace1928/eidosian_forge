from __future__ import annotations
import ast
import html
import json
import logging
import os
import pathlib
import re
import sys
import traceback
import urllib.parse as urlparse
from contextlib import contextmanager
from types import ModuleType
from typing import IO, Any, Callable
import bokeh.command.util
from bokeh.application.handlers.code import CodeHandler
from bokeh.application.handlers.code_runner import CodeRunner
from bokeh.application.handlers.handler import Handler, handle_exception
from bokeh.core.types import PathLike
from bokeh.document import Document
from bokeh.io.doc import curdoc, patch_curdoc, set_curdoc as bk_set_curdoc
from bokeh.util.dependencies import import_required
from ..config import config
from .mime_render import MIME_RENDERERS
from .profile import profile_ctx
from .reload import record_modules
from .state import state
def extract_code(filehandle: IO, supported_syntax: tuple[str, ...]=('{pyodide}', 'python')) -> str:
    """
    Extracts Panel application code from a Markdown file.
    """
    inblock = False
    block_opener = None
    title = None
    markdown = []
    out = []
    while True:
        line = filehandle.readline()
        if not line:
            break
        lsline = line.lstrip()
        if inblock:
            if lsline.startswith(block_opener):
                inblock = False
            else:
                out.append(line)
        elif lsline.startswith('```'):
            num_leading_backticks = len(lsline) - len(lsline.lstrip('`'))
            block_opener = '`' * num_leading_backticks
            syntax = line.strip()[num_leading_backticks:]
            if syntax in supported_syntax:
                if markdown:
                    md = ''.join(markdown)
                    markdown.clear()
                    if any(('pn.extension' in o for o in out)):
                        out.append(f'pn.pane.Markdown({md!r}).servable()\n')
                inblock = True
            else:
                markdown.append(line)
        elif line.startswith('# '):
            title = line[1:].lstrip()
        else:
            markdown.append(line)
    if markdown:
        md = ''.join(markdown)
        if any(('pn.extension' in o for o in out)):
            out.append(f'pn.pane.Markdown({md!r}).servable()\n')
    if title and any(('template=' in o for o in out if 'pn.extension' in o)):
        out.append(f'pn.state.template.title = {title.strip()!r}')
    return '\n'.join(out)