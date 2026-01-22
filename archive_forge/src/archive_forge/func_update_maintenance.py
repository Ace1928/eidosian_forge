from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def update_maintenance(self, maintenance_id, group_ids, host_ids, start_time, maintenance_type, period, desc, tags):
    end_time = start_time + period
    parameters = {'maintenanceid': maintenance_id, 'groupids': group_ids, 'hostids': host_ids, 'maintenance_type': maintenance_type, 'active_since': str(start_time), 'active_till': str(end_time), 'description': desc, 'timeperiods': [{'timeperiod_type': '0', 'start_date': str(start_time), 'period': str(period)}]}
    if tags is not None:
        parameters['tags'] = tags
    self._zapi.maintenance.update(parameters)
    return (0, None, None)