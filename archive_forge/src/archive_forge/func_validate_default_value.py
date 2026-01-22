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
def validate_default_value(handler, default, norm, param='value'):
    """
    assert helper that quickly validates default value.
    designed to get out of the way and reduce overhead when asserts are stripped.
    """
    assert default is not None, '%s lacks default %s' % (handler.name, param)
    assert norm(default) == default, '%s: invalid default %s: %r' % (handler.name, param, default)
    return True