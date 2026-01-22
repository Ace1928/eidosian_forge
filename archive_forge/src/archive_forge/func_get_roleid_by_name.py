from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible_collections.community.zabbix.plugins.module_utils.helpers import (
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_roleid_by_name(self, role_name):
    roles = self._zapi.role.get({'output': 'extend'})
    for role in roles:
        if role['name'] == role_name:
            return role['roleid']
    self._module.fail_json(msg='Role not found: %s' % role_name)