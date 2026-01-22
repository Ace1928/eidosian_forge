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
def patched_generate_tokens(original_tokens):
    """
    Fixes tokens yielded by `tokenize.generate_tokens` to handle more non-ASCII characters in identifiers.
    Workaround for https://github.com/python/cpython/issues/68382.
    Should only be used when tokenizing a string that is known to be valid syntax,
    because it assumes that error tokens are not actually errors.
    Combines groups of consecutive NAME, NUMBER, and/or ERRORTOKEN tokens into a single NAME token.
    """
    group = []
    for tok in original_tokens:
        if tok.type in (tokenize.NAME, tokenize.ERRORTOKEN, tokenize.NUMBER) and (not group or group[-1].end == tok.start):
            group.append(tok)
        else:
            for combined_token in combine_tokens(group):
                yield combined_token
            group = []
            yield tok
    for combined_token in combine_tokens(group):
        yield combined_token