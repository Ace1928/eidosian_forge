import sys
import re
from abc import ABCMeta
from unicodedata import name as unicode_name
from decimal import Decimal, DecimalException
from typing import Any, cast, overload, Callable, Dict, Generic, List, \
@classmethod
def nullary(cls, symbol: str, bp: int=0) -> Type[TK_co]:
    """Register a token for a symbol that represents a *nullary* operator."""

    def nud(self: Token[TK_co]) -> Token[TK_co]:
        return self
    return cls.register(symbol, label='operator', lbp=bp, nud=nud)