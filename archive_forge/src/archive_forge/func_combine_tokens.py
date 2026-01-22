import ast
import collections
import io
import sys
import token
import tokenize
from abc import ABCMeta
from ast import Module, expr, AST
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union, cast, Any, TYPE_CHECKING
from six import iteritems
def combine_tokens(group):
    if not any((tok.type == tokenize.ERRORTOKEN for tok in group)) or len({tok.line for tok in group}) != 1:
        return group
    return [tokenize.TokenInfo(type=tokenize.NAME, string=''.join((t.string for t in group)), start=group[0].start, end=group[-1].end, line=group[0].line)]