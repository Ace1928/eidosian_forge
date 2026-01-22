from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def delete_admin(meraki, org_id, admin_id):
    path = meraki.construct_path('revoke', 'admin', org_id=org_id) + admin_id
    r = meraki.request(path, method='DELETE')
    if meraki.status == 204:
        return r