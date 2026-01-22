import itertools
import re
from collections import deque
from contextlib import contextmanager
from sqlparse.compat import text_type
def imt(token, i=None, m=None, t=None):
    """Helper function to simplify comparisons Instance, Match and TokenType
    :param token:
    :param i: Class or Tuple/List of Classes
    :param m: Tuple of TokenType & Value. Can be list of Tuple for multiple
    :param t: TokenType or Tuple/List of TokenTypes
    :return:  bool
    """
    clss = i
    types = [t] if t and (not isinstance(t, list)) else t
    mpatterns = [m] if m and (not isinstance(m, list)) else m
    if token is None:
        return False
    elif clss and isinstance(token, clss):
        return True
    elif mpatterns and any((token.match(*pattern) for pattern in mpatterns)):
        return True
    elif types and any((token.ttype in ttype for ttype in types)):
        return True
    else:
        return False