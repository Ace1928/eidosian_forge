from __future__ import absolute_import, division, print_function
import json
import re
from difflib import Differ
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
def commit_config(module, comment=None, confirmed=False, confirm_timeout=None, persist=False, check=False, label=None):
    conn = get_connection(module)
    reply = None
    try:
        if is_netconf(module):
            if check:
                reply = conn.validate(remove_ns=True)
            else:
                reply = conn.commit(confirmed=confirmed, timeout=confirm_timeout, persist=persist, remove_ns=True)
        elif is_cliconf(module):
            if check:
                module.fail_json(msg='Validate configuration is not supported with network_cli connection type')
            else:
                reply = conn.commit(comment=comment, label=label)
    except ConnectionError as exc:
        module.fail_json(msg=to_text(exc, errors='surrogate_then_replace'))
    return reply