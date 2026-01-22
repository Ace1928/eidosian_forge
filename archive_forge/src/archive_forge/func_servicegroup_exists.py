from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import copy
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (ConfigProxy, get_nitro_client, netscaler_common_arguments,
def servicegroup_exists(client, module):
    log('Checking if service group exists')
    count = servicegroup.count_filtered(client, 'servicegroupname:%s' % module.params['servicegroupname'])
    log('count is %s' % count)
    if count > 0:
        return True
    else:
        return False