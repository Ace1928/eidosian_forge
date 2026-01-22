from __future__ import with_statement
import re
import logging; log = logging.getLogger(__name__)
import threading
import time
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedStringError, ExpectedTypeError, PasslibConfigWarning
from passlib.registry import get_crypt_handler, _validate_handler_name
from passlib.utils import (handlers as uh, to_bytes,
from passlib.utils.binary import BASE64_CHARS
from passlib.utils.compat import (iteritems, num_types, irange,
from passlib.utils.decor import deprecated_method, memoized_property
def get_min_verify_time(self, category=None):
    """get min_verify_time setting for policy.

        .. deprecated:: 1.6
            min_verify_time option will be removed entirely in passlib 1.8

        .. versionchanged:: 1.7
            this method now always returns the value automatically
            calculated by :meth:`CryptContext.min_verify_time`,
            any value specified by policy is ignored.
        """
    warn('get_min_verify_time() and min_verify_time option is deprecated and ignored, and will be removed in Passlib 1.8', DeprecationWarning, stacklevel=2)
    return 0