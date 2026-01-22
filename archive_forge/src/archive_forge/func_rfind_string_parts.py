from __future__ import annotations
import argparse
import io
import keyword
import re
import sys
import tokenize
from typing import Generator
from typing import Iterable
from typing import NamedTuple
from typing import Pattern
from typing import Sequence
def rfind_string_parts(tokens: Sequence[Token], i: int) -> tuple[int, ...]:
    """find the indicies of the string parts of a (joined) string literal

    - `i` should start at the end of the string literal
    - returns `()` (an empty tuple) for things which are not string literals
    """
    ret = []
    depth = 0
    for i in range(i, -1, -1):
        token = tokens[i]
        if token.name == 'STRING':
            ret.append(i)
        elif token.name in NON_CODING_TOKENS:
            pass
        elif token.src == ')':
            depth += 1
        elif depth and token.src == '(':
            depth -= 1
            if depth == 0:
                for j in range(i - 1, -1, -1):
                    tok = tokens[j]
                    if tok.name in NON_CODING_TOKENS:
                        pass
                    elif tok.src in {']', ')'} or (tok.name == 'NAME' and tok.src not in keyword.kwlist):
                        return ()
                    else:
                        break
                break
        elif depth:
            return ()
        else:
            break
    return tuple(reversed(ret))