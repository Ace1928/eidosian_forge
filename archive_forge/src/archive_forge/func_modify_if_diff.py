from __future__ import (absolute_import, division, print_function)
import json
import os
import socket
import uuid
import re
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import fetch_url, HAS_GSSAPI
from ansible.module_utils.basic import env_fallback, AnsibleFallbackNotFound
def modify_if_diff(self, name, ipa_list, module_list, add_method, remove_method, item=None, append=None):
    changed = False
    diff = list(set(ipa_list) - set(module_list))
    if append is not True and len(diff) > 0:
        changed = True
        if not self.module.check_mode:
            if item:
                remove_method(name=name, item={item: diff})
            else:
                remove_method(name=name, item=diff)
    diff = list(set(module_list) - set(ipa_list))
    if len(diff) > 0:
        changed = True
        if not self.module.check_mode:
            if item:
                add_method(name=name, item={item: diff})
            else:
                add_method(name=name, item=diff)
    return changed