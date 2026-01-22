from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_host_ids(self, host_names, zabbix_host):
    host_ids = []
    for host in host_names:
        result = self._zapi.host.get({'output': 'extend', 'filter': {zabbix_host: host}})
        if not result:
            return (1, None, 'Host id for host %s not found' % host)
        host_ids.append(result[0]['hostid'])
    return (0, host_ids, None)