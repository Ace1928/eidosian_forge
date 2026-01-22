from __future__ import annotations
from fontTools.misc.textTools import byteord, tostr
import re
from bisect import bisect_right
from typing import Literal, TypeVar, overload
from . import Blocks, Scripts, ScriptExtensions, OTTags
def script_horizontal_direction(script_code: str, default: T | type[KeyError]=KeyError) -> HorizDirection | T:
    """Return "RTL" for scripts that contain right-to-left characters
    according to the Bidi_Class property. Otherwise return "LTR".
    """
    if script_code not in Scripts.NAMES:
        if isinstance(default, type) and issubclass(default, KeyError):
            raise default(script_code)
        return default
    return 'RTL' if script_code in RTL_SCRIPTS else 'LTR'