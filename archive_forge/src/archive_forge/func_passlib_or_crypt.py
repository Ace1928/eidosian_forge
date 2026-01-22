from __future__ import (absolute_import, division, print_function)
import random
import re
import string
import sys
from collections import namedtuple
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.module_utils.six import text_type
from ansible.module_utils.common.text.converters import to_text, to_bytes
from ansible.utils.display import Display
def passlib_or_crypt(secret, algorithm, salt=None, salt_size=None, rounds=None, ident=None):
    display.deprecated('passlib_or_crypt API is deprecated in favor of do_encrypt', version='2.20')
    return do_encrypt(secret, algorithm, salt=salt, salt_size=salt_size, rounds=rounds, ident=ident)