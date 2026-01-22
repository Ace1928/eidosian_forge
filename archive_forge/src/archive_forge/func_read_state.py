from __future__ import absolute_import, division, print_function
import re
import os
import time
import tempfile
import filecmp
import shutil
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
def read_state(b_path):
    """
    Read a file and store its content in a variable as a list.
    """
    with open(b_path, 'r') as f:
        text = f.read()
    return [t for t in text.splitlines() if t != '']