from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.redis import (
import re
def set_master_mode(client):
    try:
        return client.slaveof()
    except Exception:
        return False