import regex._regex_core as _regex_core
import regex._regex as _regex
from threading import RLock as _RLock
from locale import getpreferredencoding as _getpreferredencoding
from regex._regex_core import *
from regex._regex_core import (_ALL_VERSIONS, _ALL_ENCODINGS, _FirstSetError,
from regex._regex_core import (ALNUM as _ALNUM, Info as _Info, OP as _OP, Source
import copyreg as _copy_reg
def complain_unused_args():
    if ignore_unused:
        return
    unused_kwargs = set(kwargs) - {k for k, v in args_needed}
    if unused_kwargs:
        any_one = next(iter(unused_kwargs))
        raise ValueError('unused keyword argument {!a}'.format(any_one))