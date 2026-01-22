from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def get_org_devices(meraki, org_id):
    """ Get all devices in an organization """
    path = meraki.construct_path('get_all_org', org_id=org_id)
    response = meraki.request(path, method='GET')
    if meraki.status != 200:
        meraki.fail_json(msg='Failed to query all devices belonging to the organization')
    return response