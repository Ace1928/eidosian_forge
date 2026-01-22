from __future__ import absolute_import, division, print_function
import re
import os
import time
import tempfile
import filecmp
import shutil
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
def filter_and_format_state(string):
    """
    Remove timestamps to ensure idempotence between runs. Also remove counters
    by default. And return the result as a list.
    """
    string = re.sub('((^|\\n)# (Generated|Completed)[^\\n]*) on [^\\n]*', '\\1', string)
    if not module.params['counters']:
        string = re.sub('\\[[0-9]+:[0-9]+\\]', '[0:0]', string)
    lines = [line for line in string.splitlines() if line != '']
    return lines