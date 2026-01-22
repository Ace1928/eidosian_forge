from __future__ import annotations
from collections.abc import Sequence
import inspect
from typing import Any, ClassVar, Protocol
from .common.utils import escapeHtml, unescapeAll
from .token import Token
from .utils import EnvType, OptionsDict
def code_inline(self, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType) -> str:
    token = tokens[idx]
    return '<code' + self.renderAttrs(token) + '>' + escapeHtml(tokens[idx].content) + '</code>'