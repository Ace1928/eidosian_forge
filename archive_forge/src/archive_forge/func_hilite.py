from __future__ import annotations
import functools
import json
import textwrap
from typing import (
import param  # type: ignore
from ..io.resources import CDN_DIST
from ..models import HTML as _BkHTML, JSON as _BkJSON
from ..util import HTML_SANITIZER, escape
from .base import ModelPane
def hilite(token, langname, attrs):
    try:
        from markdown.extensions.codehilite import CodeHilite
        return CodeHilite(src=token, lang=langname).hilite()
    except Exception:
        return token