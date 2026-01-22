from __future__ import annotations
import ast
import io
import os
import sys
import tokenize
from collections.abc import (
from os.path import relpath
from textwrap import dedent
from tokenize import COMMENT, NAME, OP, STRING, generate_tokens
from typing import TYPE_CHECKING, Any
from babel.util import parse_encoding, parse_future_flags, pathmatch
def extract_nothing(fileobj: _FileObj, keywords: Mapping[str, _Keyword], comment_tags: Collection[str], options: Mapping[str, Any]) -> list[_ExtractionResult]:
    """Pseudo extractor that does not actually extract anything, but simply
    returns an empty list.
    """
    return []