from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.redis import (
import re
def set_replica_mode(client, master_host, master_port):
    try:
        return client.slaveof(master_host, master_port)
    except Exception:
        return False