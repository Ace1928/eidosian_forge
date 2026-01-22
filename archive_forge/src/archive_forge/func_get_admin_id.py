from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def get_admin_id(meraki, data, name=None, email=None):
    admin_id = None
    for a in data:
        if meraki.params['name'] is not None:
            if meraki.params['name'] == a['name']:
                if admin_id is not None:
                    meraki.fail_json(msg='There are multiple administrators with the same name')
                else:
                    admin_id = a['id']
        elif meraki.params['email']:
            if meraki.params['email'] == a['email']:
                return a['id']
    if admin_id is None:
        meraki.fail_json(msg='No admin_id found')
    return admin_id