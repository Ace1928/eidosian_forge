from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.helpers import parseLinkLabel
from markdown_it.rules_block import StateBlock
from markdown_it.rules_core import StateCore
from markdown_it.rules_inline import StateInline
from markdown_it.token import Token
from mdit_py_plugins.utils import is_code_block
def footnote_inline(state: StateInline, silent: bool) -> bool:
    """Process inline footnotes (^[...])"""
    maximum = state.posMax
    start = state.pos
    if start + 2 >= maximum:
        return False
    if state.src[start] != '^':
        return False
    if state.src[start + 1] != '[':
        return False
    labelStart = start + 2
    labelEnd = parseLinkLabel(state, start + 1)
    if labelEnd < 0:
        return False
    if not silent:
        refs = state.env.setdefault('footnotes', {}).setdefault('list', {})
        footnoteId = len(refs)
        tokens: List[Token] = []
        state.md.inline.parse(state.src[labelStart:labelEnd], state.md, state.env, tokens)
        token = state.push('footnote_ref', '', 0)
        token.meta = {'id': footnoteId}
        refs[footnoteId] = {'content': state.src[labelStart:labelEnd], 'tokens': tokens}
    state.pos = labelEnd + 1
    state.posMax = maximum
    return True