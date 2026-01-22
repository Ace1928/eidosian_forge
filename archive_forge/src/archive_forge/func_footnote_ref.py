from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.helpers import parseLinkLabel
from markdown_it.rules_block import StateBlock
from markdown_it.rules_core import StateCore
from markdown_it.rules_inline import StateInline
from markdown_it.token import Token
from mdit_py_plugins.utils import is_code_block
def footnote_ref(state: StateInline, silent: bool) -> bool:
    """Process footnote references ([^...])"""
    maximum = state.posMax
    start = state.pos
    if start + 3 > maximum:
        return False
    if 'footnotes' not in state.env or 'refs' not in state.env['footnotes']:
        return False
    if state.src[start] != '[':
        return False
    if state.src[start + 1] != '^':
        return False
    pos = start + 2
    while pos < maximum:
        if state.src[pos] == ' ':
            return False
        if state.src[pos] == '\n':
            return False
        if state.src[pos] == ']':
            break
        pos += 1
    if pos == start + 2:
        return False
    if pos >= maximum:
        return False
    pos += 1
    label = state.src[start + 2:pos - 1]
    if ':' + label not in state.env['footnotes']['refs']:
        return False
    if not silent:
        if 'list' not in state.env['footnotes']:
            state.env['footnotes']['list'] = {}
        if state.env['footnotes']['refs'][':' + label] < 0:
            footnoteId = len(state.env['footnotes']['list'])
            state.env['footnotes']['list'][footnoteId] = {'label': label, 'count': 0}
            state.env['footnotes']['refs'][':' + label] = footnoteId
        else:
            footnoteId = state.env['footnotes']['refs'][':' + label]
        footnoteSubId = state.env['footnotes']['list'][footnoteId]['count']
        state.env['footnotes']['list'][footnoteId]['count'] += 1
        token = state.push('footnote_ref', '', 0)
        token.meta = {'id': footnoteId, 'subId': footnoteSubId, 'label': label}
    state.pos = pos
    state.posMax = maximum
    return True