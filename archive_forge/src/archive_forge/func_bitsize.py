from __future__ import with_statement
import inspect
import logging; log = logging.getLogger(__name__)
import math
import threading
from warnings import warn
import passlib.exc as exc, passlib.ifc as ifc
from passlib.exc import MissingBackendError, PasslibConfigWarning, \
from passlib.ifc import PasswordHash
from passlib.registry import get_crypt_handler
from passlib.utils import (
from passlib.utils.binary import (
from passlib.utils.compat import join_byte_values, irange, u, native_string_types, \
from passlib.utils.decor import classproperty, deprecated_method
@classmethod
def bitsize(cls, rounds=None, vary_rounds=0.1, **kwds):
    """[experimental method] return info about bitsizes of hash"""
    info = super(HasRounds, cls).bitsize(**kwds)
    if cls.rounds_cost != 'log2':
        import math
        if rounds is None:
            rounds = cls.default_rounds
        info['rounds'] = max(0, int(1 + math.log(rounds * vary_rounds, 2)))
    return info