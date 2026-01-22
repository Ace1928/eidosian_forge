from __future__ import annotations
import ast
import asyncio
import inspect
import textwrap
from functools import lru_cache
from inspect import signature
from itertools import groupby
from typing import (
from langchain_core.pydantic_v1 import BaseConfig, BaseModel
from langchain_core.pydantic_v1 import create_model as _create_model_base
from langchain_core.runnables.schema import StreamEvent
def indent_lines_after_first(text: str, prefix: str) -> str:
    """Indent all lines of text after the first line.

    Args:
        text:  The text to indent
        prefix: Used to determine the number of spaces to indent

    Returns:
        str: The indented text
    """
    n_spaces = len(prefix)
    spaces = ' ' * n_spaces
    lines = text.splitlines()
    return '\n'.join([lines[0]] + [spaces + line for line in lines[1:]])