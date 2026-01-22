import base64
import datetime
import re
import struct
import typing as t
import uuid
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.collections import is_sequence
@per_sequence
def dn_escape(value: str) -> str:
    """Escapes a DistinguisedName attribute value."""
    escaped_value = []
    end_idx = len(value) - 1
    for idx, c in enumerate(value):
        if idx == 0 and c in [' ', '#'] or (idx == end_idx and c == ' ') or c in ['"', '+', ',', ';', '<', '>', '\\']:
            escaped_value.append(f'\\{c}')
        elif c in ['\x00', '\n', '\r', '=', '/']:
            escaped_int = ord(c)
            escaped_value.append(f'\\{escaped_int:02X}')
        else:
            escaped_value.append(c)
    return ''.join(escaped_value)