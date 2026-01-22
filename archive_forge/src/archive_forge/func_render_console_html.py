from __future__ import annotations
import itertools
import linecache
import os
import re
import sys
import sysconfig
import traceback
import typing as t
from markupsafe import escape
from ..utils import cached_property
from .console import Console
def render_console_html(secret: str, evalex_trusted: bool) -> str:
    return CONSOLE_HTML % {'evalex': 'true', 'evalex_trusted': 'true' if evalex_trusted else 'false', 'console': 'true', 'title': 'Console', 'secret': secret}