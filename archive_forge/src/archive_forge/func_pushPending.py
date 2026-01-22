from __future__ import annotations
from collections import namedtuple
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from .._compat import DATACLASS_KWARGS
from ..common.utils import isMdAsciiPunct, isPunctChar, isWhiteSpace
from ..ruler import StateBase
from ..token import Token
from ..utils import EnvType
def pushPending(self) -> Token:
    token = Token('text', '', 0)
    token.content = self.pending
    token.level = self.pendingLevel
    self.tokens.append(token)
    self.pending = ''
    return token