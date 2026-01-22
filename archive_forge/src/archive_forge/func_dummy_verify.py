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
def dummy_verify(self, elapsed=0):
    """
        Helper that applications can call when user wasn't found,
        in order to simulate time it would take to hash a password.

        Runs verify() against a dummy hash, to simulate verification
        of a real account password.

        :param elapsed:

            .. deprecated:: 1.7.1

                this option is ignored, and will be removed in passlib 1.8.

        .. versionadded:: 1.7
        """
    self.verify(self._dummy_secret, self._dummy_hash)
    return False