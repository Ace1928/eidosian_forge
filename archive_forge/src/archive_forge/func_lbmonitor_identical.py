from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def lbmonitor_identical(client, module, lbmonitor_proxy):
    log('Checking if monitor is identical')
    count = lbmonitor.count_filtered(client, 'monitorname:%s' % module.params['monitorname'])
    if count == 0:
        return False
    lbmonitor_list = lbmonitor.get_filtered(client, 'monitorname:%s' % module.params['monitorname'])
    diff_dict = lbmonitor_proxy.diff_object(lbmonitor_list[0])
    hashed_fields = ['password', 'secondarypassword', 'radkey']
    for key in hashed_fields:
        if key in diff_dict:
            del diff_dict[key]
    if diff_dict == {}:
        return True
    else:
        return False