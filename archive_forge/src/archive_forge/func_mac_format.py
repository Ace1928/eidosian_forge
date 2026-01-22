from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
from ansible.module_utils.connection import exec_command
def mac_format(mac):
    """convert mac format to xxxx-xxxx-xxxx"""
    if not mac:
        return None
    if mac.count('-') != 2:
        return None
    addrs = mac.split('-')
    for i in range(3):
        if not addrs[i] or not addrs[i].isalnum():
            return None
        if len(addrs[i]) < 1 or len(addrs[i]) > 4:
            return None
        try:
            addrs[i] = int(addrs[i], 16)
        except ValueError:
            return None
    try:
        return '%04x-%04x-%04x' % (addrs[0], addrs[1], addrs[2])
    except ValueError:
        return None
    except TypeError:
        return None