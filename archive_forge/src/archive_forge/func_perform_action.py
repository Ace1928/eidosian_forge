from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, scaleway_argument_spec, Scaleway
def perform_action(compute_api, server, action):
    response = compute_api.post(path='servers/%s/action' % server['id'], data={'action': action})
    if not response.ok:
        msg = 'Error during server %s: (%s) %s' % (action, response.status_code, response.json)
        compute_api.module.fail_json(msg=msg)
    wait_to_complete_state_transition(compute_api=compute_api, server=server)
    return response