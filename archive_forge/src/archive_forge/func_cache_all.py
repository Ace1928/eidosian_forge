import regex._regex_core as _regex_core
import regex._regex as _regex
from threading import RLock as _RLock
from locale import getpreferredencoding as _getpreferredencoding
from regex._regex_core import *
from regex._regex_core import (_ALL_VERSIONS, _ALL_ENCODINGS, _FirstSetError,
from regex._regex_core import (ALNUM as _ALNUM, Info as _Info, OP as _OP, Source
import copyreg as _copy_reg
def cache_all(value=True):
    """Sets whether to cache all patterns, even those are compiled explicitly.
    Passing None has no effect, but returns the current setting."""
    global _cache_all
    if value is None:
        return _cache_all
    _cache_all = value