from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def inventory_mode_numeric(self, inventory_mode):
    if inventory_mode == 'automatic':
        return int(1)
    elif inventory_mode == 'manual':
        return int(0)
    elif inventory_mode == 'disabled':
        return int(-1)
    return inventory_mode