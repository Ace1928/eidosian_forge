from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def dump_services(self, service_ids):
    services = self._zapi.service.get({'output': 'extend', 'filter': {'serviceid': service_ids}, 'selectParents': 'extend', 'selectTags': 'extend', 'selectProblemTags': 'extend', 'selectChildren': 'extend', 'selectStatusRules': 'extend'})
    return services