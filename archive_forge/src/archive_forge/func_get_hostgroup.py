from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def get_hostgroup(module, array):
    hostgroup = None
    for host in array.list_hgroups():
        if host['name'].casefold() == module.params['name'].casefold():
            hostgroup = host
            break
    return hostgroup