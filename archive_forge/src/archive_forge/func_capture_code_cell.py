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
def capture_code_cell(cell):
    """
    Parses a code cell and generates wrapper code to capture the
    return value of the cell and any other outputs published inside
    the cell.
    """
    code = []
    if not len(cell['source']):
        return code
    source = cell['source'].split('\n')
    for line in source[:-1]:
        line = line.replace('get_ipython().run_line_magic', '').replace('get_ipython().magic', '')
        code.append(line)
    cell_out = source[-1]
    parses = False
    while not parses:
        try:
            ast.parse(cell_out)
            parses = True
        except SyntaxError:
            if not code:
                break
            cell_out = f'{code.pop()}\n{cell_out}'
    if not parses:
        log.warn(f'The following cell did not contain valid Python syntax and was skipped:\n\n{cell['source']}')
        return code
    elif cell_out.rstrip().endswith(';'):
        code.append(cell_out)
        return code
    if '#' in cell_out:
        try:
            cell_tmp = cell_out[:cell_out.index('#')].rstrip()
            ast.parse(cell_tmp)
            cell_out = cell_tmp
        except SyntaxError:
            pass
    try:
        ast.parse(cell_out, mode='eval')
    except SyntaxError:
        code.append(cell_out)
        return code
    cell_id = cell['id']
    code.append(f'_pn__state._cell_outputs[{cell_id!r}].append(({cell_out}))\nfor _cell__out in _CELL__DISPLAY:\n    _pn__state._cell_outputs[{cell_id!r}].append(_cell__out)\n_CELL__DISPLAY.clear()\n_fig__out = _get__figure()\nif _fig__out:\n    _pn__state._cell_outputs[{cell_id!r}].append(_fig__out)\n')
    return code