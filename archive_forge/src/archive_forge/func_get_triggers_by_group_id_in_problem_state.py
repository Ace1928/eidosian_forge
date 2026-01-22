from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_triggers_by_group_id_in_problem_state(self, group_id, trigger_severity):
    """ Get triggers in problem state from a groupid"""
    output = 'extend'
    triggers_list = self._zapi.trigger.get({'output': output, 'groupids': group_id, 'min_severity': trigger_severity})
    return triggers_list