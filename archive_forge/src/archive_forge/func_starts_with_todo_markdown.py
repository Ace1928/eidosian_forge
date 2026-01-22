from __future__ import annotations
import re
from uuid import uuid4
from markdown_it import MarkdownIt
from markdown_it.rules_core import StateCore
from markdown_it.token import Token
def starts_with_todo_markdown(token: Token) -> bool:
    return re.match(f'\\[[ xX]]{_GFM_WHITESPACE_RE}+', token.content) is not None