from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_snmp_traps(group, module):
    body = execute_show_command('show run snmp all', module)[0].split('\n')
    resource = {}
    feature_list = ['aaa', 'bfd', 'bgp', 'bridge', 'callhome', 'cfs', 'config', 'eigrp', 'entity', 'feature-control', 'generic', 'hsrp', 'license', 'link', 'lldp', 'mmode', 'ospf', 'pim', 'rf', 'rmon', 'snmp', 'storm-control', 'stpx', 'switchfabric', 'syslog', 'sysmgr', 'system', 'upgrade', 'vtp']
    if 'all' in group and 'N3K-C35' in get_platform_id(module):
        module.warn("Platform does not support bfd traps; bfd ignored for 'group: all' request")
        feature_list.remove('bfd')
    for each in feature_list:
        for line in body:
            if each == 'ospf':
                if 'snmp-server enable traps ospf' == line:
                    resource[each] = True
                    break
            elif 'enable traps {0}'.format(each) in line:
                if 'no ' in line:
                    resource[each] = False
                    break
                else:
                    resource[each] = True
    for each in feature_list:
        if resource.get(each) is None:
            body = execute_show_command('show run | inc feature', module)[0]
            if 'feature {0}'.format(each) in body:
                resource[each] = False
    find = resource.get(group, None)
    if group == 'all'.lower():
        return resource
    elif find is not None:
        trap_resource = {group: find}
        return trap_resource
    else:
        return {}