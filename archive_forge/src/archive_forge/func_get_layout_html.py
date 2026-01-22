from __future__ import annotations
import logging # isort:skip
import io
import os
from contextlib import contextmanager
from os.path import abspath, expanduser, splitext
from tempfile import mkstemp
from typing import (
from ..core.types import PathLike
from ..document import Document
from ..embed import file_html
from ..resources import INLINE, Resources
from ..themes import Theme
from ..util.warnings import warn
from .state import State, curstate
from .util import default_filename
def get_layout_html(obj: UIElement | Document, *, resources: Resources=INLINE, width: int | None=None, height: int | None=None, theme: Theme | None=None) -> str:
    """

    """
    template = '\\\n    {% block preamble %}\n    <style>\n        html, body {\n            box-sizing: border-box;\n            width: 100%;\n            height: 100%;\n            margin: 0;\n            border: 0;\n            padding: 0;\n            overflow: hidden;\n        }\n    </style>\n    {% endblock %}\n    '

    def html() -> str:
        return file_html(obj, resources=resources, title='', template=template, theme=theme, suppress_callback_warning=True, _always_new=True)
    if width is not None or height is not None:
        from ..models.plots import Plot
        if not isinstance(obj, Plot):
            warn('Export method called with width or height argument on a non-Plot model. The size values will be ignored.')
        else:
            with _resized(obj, width, height):
                return html()
    return html()