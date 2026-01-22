from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def gslb_vserver_identical(client, module, gslb_vserver_proxy):
    gslb_vserver_list = gslbvserver.get_filtered(client, 'name:%s' % module.params['name'])
    diff_dict = gslb_vserver_proxy.diff_object(gslb_vserver_list[0])
    if len(diff_dict) != 0:
        return False
    else:
        return True