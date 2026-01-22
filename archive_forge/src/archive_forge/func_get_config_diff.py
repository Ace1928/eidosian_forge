from __future__ import absolute_import, division, print_function
import json
import re
from difflib import Differ
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
def get_config_diff(module, running=None, candidate=None):
    conn = get_connection(module)
    if is_cliconf(module):
        try:
            response = conn.get('show commit changes diff')
        except ConnectionError as exc:
            module.fail_json(msg=to_text(exc, errors='surrogate_then_replace'))
        return response
    elif is_netconf(module):
        if running and candidate:
            running_data_ele = etree.fromstring(to_bytes(running.strip())).getchildren()[0]
            candidate_data_ele = etree.fromstring(to_bytes(candidate.strip())).getchildren()[0]
            running_data = to_text(etree.tostring(running_data_ele)).strip()
            candidate_data = to_text(etree.tostring(candidate_data_ele)).strip()
            if running_data != candidate_data:
                d = Differ()
                diff = list(d.compare(running_data.splitlines(), candidate_data.splitlines()))
                return '\n'.join(diff).strip()
    return None