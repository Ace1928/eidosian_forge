from __future__ import annotations
from warnings import warn
import ast
import codeop
import io
import re
import sys
import tokenize
import warnings
from typing import List, Tuple, Union, Optional, TYPE_CHECKING
from types import CodeType
from IPython.core.inputtransformer import (leading_indent,
from IPython.utils import tokenutil
from IPython.core.inputtransformer import (ESC_SHELL, ESC_SH_CAP, ESC_HELP,
def find_next_indent(code) -> int:
    """Find the number of spaces for the next line of indentation"""
    tokens = list(partial_tokens(code))
    if tokens[-1].type == tokenize.ENDMARKER:
        tokens.pop()
    if not tokens:
        return 0
    while tokens[-1].type in {tokenize.DEDENT, tokenize.NEWLINE, tokenize.COMMENT, tokenize.ERRORTOKEN}:
        tokens.pop()
    if tokens[-1].type == IN_MULTILINE_STATEMENT:
        while tokens[-2].type in {tokenize.NL}:
            tokens.pop(-2)
    if tokens[-1].type == INCOMPLETE_STRING:
        return 0
    prev_indents = [0]

    def _add_indent(n):
        if n != prev_indents[-1]:
            prev_indents.append(n)
    tokiter = iter(tokens)
    for tok in tokiter:
        if tok.type in {tokenize.INDENT, tokenize.DEDENT}:
            _add_indent(tok.end[1])
        elif tok.type == tokenize.NL:
            try:
                _add_indent(next(tokiter).start[1])
            except StopIteration:
                break
    last_indent = prev_indents.pop()
    if tokens[-1].type == IN_MULTILINE_STATEMENT:
        if tokens[-2].exact_type in {tokenize.LPAR, tokenize.LSQB, tokenize.LBRACE}:
            return last_indent + 4
        return last_indent
    if tokens[-1].exact_type == tokenize.COLON:
        return last_indent + 4
    if last_indent:
        last_line_starts = 0
        for i, tok in enumerate(tokens):
            if tok.type == tokenize.NEWLINE:
                last_line_starts = i + 1
        last_line_tokens = tokens[last_line_starts:]
        names = [t.string for t in last_line_tokens if t.type == tokenize.NAME]
        if names and names[0] in {'raise', 'return', 'pass', 'break', 'continue'}:
            for indent in reversed(prev_indents):
                if indent < last_indent:
                    return indent
    return last_indent