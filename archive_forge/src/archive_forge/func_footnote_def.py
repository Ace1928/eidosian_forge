from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.helpers import parseLinkLabel
from markdown_it.rules_block import StateBlock
from markdown_it.rules_core import StateCore
from markdown_it.rules_inline import StateInline
from markdown_it.token import Token
from mdit_py_plugins.utils import is_code_block
def footnote_def(state: StateBlock, startLine: int, endLine: int, silent: bool) -> bool:
    """Process footnote block definition"""
    if is_code_block(state, startLine):
        return False
    start = state.bMarks[startLine] + state.tShift[startLine]
    maximum = state.eMarks[startLine]
    if start + 4 > maximum:
        return False
    if state.src[start] != '[':
        return False
    if state.src[start + 1] != '^':
        return False
    pos = start + 2
    while pos < maximum:
        if state.src[pos] == ' ':
            return False
        if state.src[pos] == ']':
            break
        pos += 1
    if pos == start + 2:
        return False
    pos += 1
    if pos >= maximum or state.src[pos] != ':':
        return False
    if silent:
        return True
    pos += 1
    label = state.src[start + 2:pos - 2]
    state.env.setdefault('footnotes', {}).setdefault('refs', {})[':' + label] = -1
    open_token = Token('footnote_reference_open', '', 1)
    open_token.meta = {'label': label}
    open_token.level = state.level
    state.level += 1
    state.tokens.append(open_token)
    oldBMark = state.bMarks[startLine]
    oldTShift = state.tShift[startLine]
    oldSCount = state.sCount[startLine]
    oldParentType = state.parentType
    posAfterColon = pos
    initial = offset = state.sCount[startLine] + pos - (state.bMarks[startLine] + state.tShift[startLine])
    while pos < maximum:
        ch = state.src[pos]
        if ch == '\t':
            offset += 4 - offset % 4
        elif ch == ' ':
            offset += 1
        else:
            break
        pos += 1
    state.tShift[startLine] = pos - posAfterColon
    state.sCount[startLine] = offset - initial
    state.bMarks[startLine] = posAfterColon
    state.blkIndent += 4
    state.parentType = 'footnote'
    if state.sCount[startLine] < state.blkIndent:
        state.sCount[startLine] += state.blkIndent
    state.md.block.tokenize(state, startLine, endLine)
    state.parentType = oldParentType
    state.blkIndent -= 4
    state.tShift[startLine] = oldTShift
    state.sCount[startLine] = oldSCount
    state.bMarks[startLine] = oldBMark
    open_token.map = [startLine, state.line]
    token = Token('footnote_reference_close', '', -1)
    state.level -= 1
    token.level = state.level
    state.tokens.append(token)
    return True