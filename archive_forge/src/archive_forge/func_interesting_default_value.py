from __future__ import annotations
import typing as t
from collections import defaultdict
from textwrap import dedent
from traitlets import HasTraits, Undefined
from traitlets.config.application import Application
from traitlets.utils.text import indent
def interesting_default_value(dv: t.Any) -> bool:
    if dv is None or dv is Undefined:
        return False
    if isinstance(dv, (str, list, tuple, dict, set)):
        return bool(dv)
    return True