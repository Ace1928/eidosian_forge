import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
@functools.wraps(func)
def transformer_factory(**kwargs):
    return cls(func, **kwargs)