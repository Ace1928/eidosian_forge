from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def wait_check_action(self, droplet_id, action_id):
    end_time = time.monotonic() + self.wait_timeout
    while time.monotonic() < end_time:
        response = self.rest.get('droplets/{0}/actions/{1}'.format(droplet_id, action_id))
        json_data = response.json
        status_code = response.status_code
        message = json_data.get('message', 'no error message')
        action = json_data.get('action', None)
        action_id = action.get('id', None)
        action_status = action.get('status', None)
        if action is None or action_id is None or action_status is None:
            self.module.fail_json(changed=False, msg=DODroplet.failure_message['unexpected'].format('no action, ID, or status'))
        if status_code >= 400:
            self.module.fail_json(changed=False, msg=DODroplet.failure_message['failed_to'].format('get', 'action', status_code, message))
        if action_status == 'errored':
            self.module.fail_json(changed=True, msg=DODroplet.failure_message['support_action'].format(action_id))
        if action_status == 'completed':
            return
        time.sleep(self.sleep_interval)
    self.module.fail_json(msg='Wait for Droplet action timeout')