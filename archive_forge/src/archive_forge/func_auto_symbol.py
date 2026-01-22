from tokenize import (generate_tokens, untokenize, TokenError,
from keyword import iskeyword
import ast
import unicodedata
from io import StringIO
import builtins
import types
from typing import Tuple as tTuple, Dict as tDict, Any, Callable, \
from sympy.assumptions.ask import AssumptionKeys
from sympy.core.basic import Basic
from sympy.core import Symbol
from sympy.core.function import Function
from sympy.utilities.misc import func_name
from sympy.functions.elementary.miscellaneous import Max, Min
def auto_symbol(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """Inserts calls to ``Symbol``/``Function`` for undefined variables."""
    result: List[TOKEN] = []
    prevTok = (-1, '')
    tokens.append((-1, ''))
    for tok, nextTok in zip(tokens, tokens[1:]):
        tokNum, tokVal = tok
        nextTokNum, nextTokVal = nextTok
        if tokNum == NAME:
            name = tokVal
            if name in ['True', 'False', 'None'] or iskeyword(name) or (prevTok[0] == OP and prevTok[1] == '.') or (prevTok[0] == OP and prevTok[1] in ('(', ',') and (nextTokNum == OP) and (nextTokVal == '=')) or (name in local_dict and local_dict[name] is not null):
                result.append((NAME, name))
                continue
            elif name in local_dict:
                local_dict.setdefault(null, set()).add(name)
                if nextTokVal == '(':
                    local_dict[name] = Function(name)
                else:
                    local_dict[name] = Symbol(name)
                result.append((NAME, name))
                continue
            elif name in global_dict:
                obj = global_dict[name]
                if isinstance(obj, (AssumptionKeys, Basic, type)) or callable(obj):
                    result.append((NAME, name))
                    continue
            result.extend([(NAME, 'Symbol' if nextTokVal != '(' else 'Function'), (OP, '('), (NAME, repr(str(name))), (OP, ')')])
        else:
            result.append((tokNum, tokVal))
        prevTok = (tokNum, tokVal)
    return result