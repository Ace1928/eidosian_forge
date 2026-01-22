import sys
import re
from abc import ABCMeta
from unicodedata import name as unicode_name
from decimal import Decimal, DecimalException
from typing import Any, cast, overload, Callable, Dict, Generic, List, \
@classmethod
def infixr(cls, symbol: str, bp: int=0) -> Type[TK_co]:
    """Register a token for a symbol that represents an *infixr* binary operator."""

    def led(self: Token[TK_co], left: Token[TK_co]) -> Token[TK_co]:
        self[:] = (left, self.parser.expression(rbp=bp - 1))
        return self
    return cls.register(symbol, label='operator', lbp=bp, rbp=bp - 1, led=led)