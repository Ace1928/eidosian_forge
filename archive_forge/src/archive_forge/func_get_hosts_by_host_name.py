from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_hosts_by_host_name(self, host_name, exact_match, host_inventory):
    """ Get host by host name """
    search_key = 'search'
    if exact_match:
        search_key = 'filter'
    host_list = self._zapi.host.get({'output': 'extend', 'selectParentTemplates': ['name'], search_key: {'host': [host_name]}, 'selectInventory': host_inventory, 'selectGroups': 'extend', 'selectTags': 'extend', 'selectMacros': 'extend'})
    if len(host_list) < 1:
        self._module.fail_json(msg='Host not found: %s' % host_name)
    else:
        return host_list