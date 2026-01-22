from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def gslb_service_exists(client, module):
    if gslbservice.count_filtered(client, 'servicename:%s' % module.params['servicename']) > 0:
        return True
    else:
        return False