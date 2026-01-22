from itertools import groupby
import numpy as np
import param
import pyparsing as pp
from ..core.options import Cycle, Options, Palette
from ..core.util import merge_option_dicts
from ..operation import Compositor
from .transform import dim
@classmethod
def collect_tokens(cls, parseresult, mode):
    """
        Collect the tokens from a (potentially) nested parse result.
        """
    inner = '(%s)' if mode == 'parens' else '[%s]'
    if parseresult is None:
        return []
    tokens = []
    for token in parseresult.asList():
        if isinstance(token, list):
            token = cls.recurse_token(token, inner)
            tokens[-1] = tokens[-1] + token
        else:
            if token.strip() == ',':
                continue
            tokens.append(cls._strip_commas(token))
    return tokens