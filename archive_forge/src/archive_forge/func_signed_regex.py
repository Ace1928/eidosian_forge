from __future__ import annotations
import re
import typing as t
import uuid
from urllib.parse import quote
@property
def signed_regex(self) -> str:
    return f'-?{self.regex}'