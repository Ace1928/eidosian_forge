from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_user_role(self, name):
    result = self._zapi.role.get({'output': 'extend', 'selectRules': 'extend', 'filter': {'name': name}})
    return result