from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_host_templates_by_host_id(self, host_id):
    template_ids = []
    template_list = self._zapi.template.get({'output': 'extend', 'hostids': host_id})
    for template in template_list:
        template_ids.append(template['templateid'])
    return template_ids