from __future__ import annotations
from collections.abc import Sequence
import inspect
from typing import Any, ClassVar, Protocol
from .common.utils import escapeHtml, unescapeAll
from .token import Token
from .utils import EnvType, OptionsDict
def renderToken(self, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType) -> str:
    """Default token renderer.

        Can be overridden by custom function

        :param idx: token index to render
        :param options: params of parser instance
        """
    result = ''
    needLf = False
    token = tokens[idx]
    if token.hidden:
        return ''
    if token.block and token.nesting != -1 and idx and tokens[idx - 1].hidden:
        result += '\n'
    result += ('</' if token.nesting == -1 else '<') + token.tag
    result += self.renderAttrs(token)
    if token.nesting == 0 and options['xhtmlOut']:
        result += ' /'
    if token.block:
        needLf = True
        if token.nesting == 1 and idx + 1 < len(tokens):
            nextToken = tokens[idx + 1]
            if nextToken.type == 'inline' or nextToken.hidden:
                needLf = False
            elif nextToken.nesting == -1 and nextToken.tag == token.tag:
                needLf = False
    result += '>\n' if needLf else '>'
    return result