from __future__ import (absolute_import, division, print_function)
import base64
import random
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import PY2
def generate_insecure_key():
    """Do NOT use this for cryptographic purposes!"""
    while True:
        if PY2:
            key = chr(random.randint(0, 255))
        else:
            key = bytes([random.randint(0, 255)])
        if key != b'\x00':
            return key