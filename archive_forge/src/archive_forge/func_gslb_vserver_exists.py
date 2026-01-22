from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def gslb_vserver_exists(client, module):
    if gslbvserver.count_filtered(client, 'name:%s' % module.params['name']) > 0:
        return True
    else:
        return False