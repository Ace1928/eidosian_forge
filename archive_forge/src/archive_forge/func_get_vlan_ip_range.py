from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def get_vlan_ip_range(self):
    if not self.ip_range:
        args = {'zoneid': self.get_zone(key='id'), 'projectid': self.get_project(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'networkid': self.get_network(key='id')}
        res = self.query_api('listVlanIpRanges', **args)
        if res:
            ip_range_list = res['vlaniprange']
            params = {'startip': self.module.params.get('start_ip'), 'endip': self.get_or_fallback('end_ip', 'start_ip')}
            for ipr in ip_range_list:
                if params['startip'] == ipr['startip'] and params['endip'] == ipr['endip']:
                    self.ip_range = ipr
                    break
    return self.ip_range