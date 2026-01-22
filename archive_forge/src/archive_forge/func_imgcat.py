from __future__ import annotations
import base64
import os
import platform
import sys
from functools import reduce
from typing import Any
def imgcat(path: str, inline: int=1, preserve_aspect_ratio: int=0, **kwargs: Any) -> str:
    return '\n%s1337;File=inline=%d;preserveAspectRatio=%d:%s%s' % (_IMG_PRE, inline, preserve_aspect_ratio, _read_as_base64(path), _IMG_POST)