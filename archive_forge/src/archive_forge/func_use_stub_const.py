from __future__ import division
import hashlib
import logging; log = logging.getLogger(__name__)
import re
import os
from struct import Struct
from warnings import warn
from passlib import exc
from passlib.utils import join_bytes, to_native_str, join_byte_values, to_bytes, \
from passlib.utils.compat import irange, int_types, unicode_or_bytes_types, PY3, error_from
from passlib.utils.decor import memoized_property
def use_stub_const(msg):
    """
            helper that installs stub constructor which throws specified error <msg>.
            """

    def const(source=b''):
        raise exc.UnknownHashError(msg, name)
    if required:
        const()
        assert "shouldn't get here"
    self.error_text = msg
    self.const = const
    try:
        self.digest_size, self.block_size = _fallback_info[name]
    except KeyError:
        pass