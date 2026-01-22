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
def has_schemes(self):
    """return True if policy defines *any* schemes for use.

        .. deprecated:: 1.6
            applications should use ``bool(context.schemes())`` instead.
            see :meth:`CryptContext.schemes`.
        """
    if self._stub_policy:
        warn(_preamble + 'Instead of ``context.policy.has_schemes()``, use ``bool(context.schemes())``.', DeprecationWarning, stacklevel=2)
    else:
        warn(_preamble + 'Instead of ``CryptPolicy().has_schemes()``, create a CryptContext instance and use ``bool(context.schemes())``.', DeprecationWarning, stacklevel=2)
    return bool(self._context.schemes())