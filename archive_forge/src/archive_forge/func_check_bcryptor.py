from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import warnings
from passlib import hash
from passlib.handlers.bcrypt import IDENT_2, IDENT_2X
from passlib.utils import repeat_string, to_bytes, is_safe_crypt_input
from passlib.utils.compat import irange, PY3
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE
def check_bcryptor(secret, hash):
    """bcryptor"""
    secret = to_native_str(secret, self.FuzzHashGenerator.password_encoding)
    if hash.startswith((IDENT_2B, IDENT_2Y)):
        hash = IDENT_2A + hash[4:]
    elif hash.startswith(IDENT_2):
        hash = IDENT_2A + hash[3:]
        if secret:
            secret = repeat_string(secret, 72)
    return Engine(False).hash_key(secret, hash) == hash