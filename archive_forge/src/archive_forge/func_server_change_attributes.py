from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, scaleway_argument_spec, Scaleway
def server_change_attributes(compute_api, target_server, wished_server):
    compute_api.module.debug('Starting patching server attributes')
    patch_payload = dict()
    for key in PATCH_MUTABLE_SERVER_ATTRIBUTES:
        if key in target_server and key in wished_server:
            if isinstance(target_server[key], dict) and 'id' in target_server[key] and wished_server[key]:
                key_dict = dict(((x, target_server[key][x]) for x in target_server[key].keys() if x != 'id'))
                key_dict['id'] = wished_server[key]
                patch_payload[key] = key_dict
            elif not isinstance(target_server[key], dict):
                patch_payload[key] = wished_server[key]
    response = compute_api.patch(path='servers/%s' % target_server['id'], data=patch_payload)
    if not response.ok:
        msg = 'Error during server attributes patching: (%s) %s' % (response.status_code, response.json)
        compute_api.module.fail_json(msg=msg)
    try:
        target_server = response.json['server']
    except KeyError:
        compute_api.module.fail_json(msg='Error in getting the server information from: %s' % response.json)
    wait_to_complete_state_transition(compute_api=compute_api, server=target_server)
    return target_server