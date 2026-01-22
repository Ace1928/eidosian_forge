from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes, tostr
from fontTools.misc import eexec
from .psOperators import (
import re
from collections.abc import Callable
from string import whitespace
import logging
def do_token(self, token, int=int, float=float, ps_name=ps_name, ps_integer=ps_integer, ps_real=ps_real):
    try:
        num = int(token)
    except (ValueError, OverflowError):
        try:
            num = float(token)
        except (ValueError, OverflowError):
            if '#' in token:
                hashpos = token.find('#')
                try:
                    base = int(token[:hashpos])
                    num = int(token[hashpos + 1:], base)
                except (ValueError, OverflowError):
                    return ps_name(token)
                else:
                    return ps_integer(num)
            else:
                return ps_name(token)
        else:
            return ps_real(num)
    else:
        return ps_integer(num)