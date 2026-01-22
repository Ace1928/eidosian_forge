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
def repeated_decimals(tokens: List[TOKEN], local_dict: DICT, global_dict: DICT):
    """
    Allows 0.2[1] notation to represent the repeated decimal 0.2111... (19/90)

    Run this before auto_number.

    """
    result: List[TOKEN] = []

    def is_digit(s):
        return all((i in '0123456789_' for i in s))
    num: List[TOKEN] = []
    for toknum, tokval in tokens:
        if toknum == NUMBER:
            if not num and '.' in tokval and ('e' not in tokval.lower()) and ('j' not in tokval.lower()):
                num.append((toknum, tokval))
            elif is_digit(tokval) and len(num) == 2:
                num.append((toknum, tokval))
            elif is_digit(tokval) and len(num) == 3 and is_digit(num[-1][1]):
                num.append((toknum, tokval))
            else:
                num = []
        elif toknum == OP:
            if tokval == '[' and len(num) == 1:
                num.append((OP, tokval))
            elif tokval == ']' and len(num) >= 3:
                num.append((OP, tokval))
            elif tokval == '.' and (not num):
                num.append((NUMBER, '0.'))
            else:
                num = []
        else:
            num = []
        result.append((toknum, tokval))
        if num and num[-1][1] == ']':
            result = result[:-len(num)]
            pre, post = num[0][1].split('.')
            repetend = num[2][1]
            if len(num) == 5:
                repetend += num[3][1]
            pre = pre.replace('_', '')
            post = post.replace('_', '')
            repetend = repetend.replace('_', '')
            zeros = '0' * len(post)
            post, repetends = [w.lstrip('0') for w in [post, repetend]]
            a = pre or '0'
            b, c = (post or '0', '1' + zeros)
            d, e = (repetends, '9' * len(repetend) + zeros)
            seq = [(OP, '('), (NAME, 'Integer'), (OP, '('), (NUMBER, a), (OP, ')'), (OP, '+'), (NAME, 'Rational'), (OP, '('), (NUMBER, b), (OP, ','), (NUMBER, c), (OP, ')'), (OP, '+'), (NAME, 'Rational'), (OP, '('), (NUMBER, d), (OP, ','), (NUMBER, e), (OP, ')'), (OP, ')')]
            result.extend(seq)
            num = []
    return result