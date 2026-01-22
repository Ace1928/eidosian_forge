from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (ConfigProxy, get_nitro_client,
def server_exists(client, module):
    log('Checking if server exists')
    if server.count_filtered(client, 'name:%s' % module.params['name']) > 0:
        return True
    else:
        return False