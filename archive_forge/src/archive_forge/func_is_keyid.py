from __future__ import absolute_import, division, print_function
import re
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native
def is_keyid(self, keystr):
    """Verifies if a key, as provided by the user is a keyid"""
    return re.match('(0x)?[0-9a-f]{8}', keystr, flags=re.IGNORECASE)