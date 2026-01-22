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
def get_context_option_with_flag(self, category, key):
    """return value of specific option, handling category inheritance.
        also returns flag indicating whether value is category-specific.
        """
    try:
        category_map = self._context_options[key]
    except KeyError:
        return (None, False)
    value = category_map.get(None)
    if category:
        try:
            alt = category_map[category]
        except KeyError:
            pass
        else:
            if value is None or alt != value:
                return (alt, True)
    return (value, False)