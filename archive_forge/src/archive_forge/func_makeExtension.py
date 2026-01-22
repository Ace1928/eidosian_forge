from __future__ import annotations
from . import Extension
from ..blockprocessors import OListProcessor, UListProcessor
import re
from typing import TYPE_CHECKING
def makeExtension(**kwargs):
    return SaneListExtension(**kwargs)