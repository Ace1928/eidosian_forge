import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
def has_comment(src):
    """Indicate whether an input line has (i.e. ends in, or is) a comment.

    This uses tokenize, so it can distinguish comments from # inside strings.

    Parameters
    ----------
    src : string
        A single line input string.

    Returns
    -------
    comment : bool
        True if source has a comment.
    """
    return tokenize.COMMENT in _line_tokens(src)