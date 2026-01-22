from __future__ import absolute_import, division, print_function
import base64
import errno
import hashlib
import hmac
import os
import os.path
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
def compute_diff(path, found_line, replace_or_add, state, key):
    diff = {'before_header': path, 'after_header': path, 'before': '', 'after': ''}
    try:
        inf = open(path, 'r')
    except IOError as e:
        if e.errno == errno.ENOENT:
            diff['before_header'] = '/dev/null'
    else:
        diff['before'] = inf.read()
        inf.close()
    lines = diff['before'].splitlines(1)
    if (replace_or_add or state == 'absent') and found_line is not None and (1 <= found_line <= len(lines)):
        del lines[found_line - 1]
    if state == 'present' and (replace_or_add or found_line is None):
        lines.append(key)
    diff['after'] = ''.join(lines)
    return diff