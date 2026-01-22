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
def extract_python(fileobj: IO[bytes], keywords: Mapping[str, _Keyword], comment_tags: Collection[str], options: _PyOptions) -> Generator[_ExtractionResult, None, None]:
    """Extract messages from Python source code.

    It returns an iterator yielding tuples in the following form ``(lineno,
    funcname, message, comments)``.

    :param fileobj: the seekable, file-like object the messages should be
                    extracted from
    :param keywords: a list of keywords (i.e. function names) that should be
                     recognized as translation functions
    :param comment_tags: a list of translator tags to search for and include
                         in the results
    :param options: a dictionary of additional options (optional)
    :rtype: ``iterator``
    """
    funcname = lineno = message_lineno = None
    call_stack = -1
    buf = []
    messages = []
    translator_comments = []
    in_def = in_translator_comments = False
    comment_tag = None
    encoding = parse_encoding(fileobj) or options.get('encoding', 'UTF-8')
    future_flags = parse_future_flags(fileobj, encoding)
    next_line = lambda: fileobj.readline().decode(encoding)
    tokens = generate_tokens(next_line)
    current_fstring_start = None
    for tok, value, (lineno, _), _, _ in tokens:
        if call_stack == -1 and tok == NAME and (value in ('def', 'class')):
            in_def = True
        elif tok == OP and value == '(':
            if in_def:
                in_def = False
                continue
            if funcname:
                message_lineno = lineno
                call_stack += 1
        elif in_def and tok == OP and (value == ':'):
            in_def = False
            continue
        elif call_stack == -1 and tok == COMMENT:
            value = value[1:].strip()
            if in_translator_comments and translator_comments[-1][0] == lineno - 1:
                translator_comments.append((lineno, value))
                continue
            for comment_tag in comment_tags:
                if value.startswith(comment_tag):
                    in_translator_comments = True
                    translator_comments.append((lineno, value))
                    break
        elif funcname and call_stack == 0:
            nested = tok == NAME and value in keywords
            if tok == OP and value == ')' or nested:
                if buf:
                    messages.append(''.join(buf))
                    del buf[:]
                else:
                    messages.append(None)
                messages = tuple(messages) if len(messages) > 1 else messages[0]
                if translator_comments and translator_comments[-1][0] < message_lineno - 1:
                    translator_comments = []
                yield (message_lineno, funcname, messages, [comment[1] for comment in translator_comments])
                funcname = lineno = message_lineno = None
                call_stack = -1
                messages = []
                translator_comments = []
                in_translator_comments = False
                if nested:
                    funcname = value
            elif tok == STRING:
                val = _parse_python_string(value, encoding, future_flags)
                if val is not None:
                    buf.append(val)
            elif tok == FSTRING_START:
                current_fstring_start = value
            elif tok == FSTRING_MIDDLE:
                if current_fstring_start is not None:
                    current_fstring_start += value
            elif tok == FSTRING_END:
                if current_fstring_start is not None:
                    fstring = current_fstring_start + value
                    val = _parse_python_string(fstring, encoding, future_flags)
                    if val is not None:
                        buf.append(val)
            elif tok == OP and value == ',':
                if buf:
                    messages.append(''.join(buf))
                    del buf[:]
                else:
                    messages.append(None)
                if translator_comments:
                    old_lineno, old_comment = translator_comments.pop()
                    translator_comments.append((old_lineno + 1, old_comment))
        elif call_stack > 0 and tok == OP and (value == ')'):
            call_stack -= 1
        elif funcname and call_stack == -1:
            funcname = None
        elif tok == NAME and value in keywords:
            funcname = value
        if current_fstring_start is not None and tok not in {FSTRING_START, FSTRING_MIDDLE}:
            current_fstring_start = None