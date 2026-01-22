from __future__ import annotations
from typing import TYPE_CHECKING
def jpeg(b: bytes):
    display_jpeg(Image(b, format='jpeg'))