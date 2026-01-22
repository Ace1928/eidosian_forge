from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
import copy
def lb_vserver_diff(client, module, lbvserver_proxy):
    lbvserver_list = lbvserver.get_filtered(client, 'name:%s' % module.params['name'])
    return lbvserver_proxy.diff_object(lbvserver_list[0])