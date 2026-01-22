from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_triggers_by_host_id_in_problem_state(self, host_id, trigger_severity):
    """ Get triggers in problem state from a hostid"""
    output = 'extend'
    triggers_list = self._zapi.trigger.get({'output': output, 'hostids': host_id, 'min_severity': trigger_severity})
    return triggers_list