from __future__ import annotations
import re
import typing
import warnings
import wcwidth
def set_byte_encoding(enc: Literal['utf8', 'narrow', 'wide']) -> None:
    if enc not in {'utf8', 'narrow', 'wide'}:
        raise ValueError(enc)
    global _byte_encoding
    _byte_encoding = enc