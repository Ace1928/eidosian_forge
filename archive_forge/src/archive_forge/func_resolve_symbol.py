from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def resolve_symbol(self, symbol, bCaseSensitive=False):
    """
        Resolves a debugging symbol's address.

        @type  symbol: str
        @param symbol: Name of the symbol to resolve.

        @type  bCaseSensitive: bool
        @param bCaseSensitive: C{True} for case sensitive matches,
            C{False} for case insensitive.

        @rtype:  int or None
        @return: Memory address of symbol. C{None} if not found.
        """
    if bCaseSensitive:
        for SymbolName, SymbolAddress, SymbolSize in self.iter_symbols():
            if symbol == SymbolName:
                return SymbolAddress
    else:
        symbol = symbol.lower()
        for SymbolName, SymbolAddress, SymbolSize in self.iter_symbols():
            if symbol == SymbolName.lower():
                return SymbolAddress