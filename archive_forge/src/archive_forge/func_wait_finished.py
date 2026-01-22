from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def wait_finished(self):
    current_time = time.monotonic()
    end_time = current_time + self.wait_timeout
    while current_time < end_time:
        response = self.rest.get('actions/{0}'.format(str(self.action_id)))
        status = response.status_code
        if status != 200:
            self.module.fail_json(msg='Unable to find action {0}, please file a bug'.format(str(self.action_id)))
        json = response.json
        if json['action']['status'] == 'completed':
            return json
        time.sleep(10)
    self.module.fail_json(msg='Timed out waiting for snapshot, action {0}'.format(str(self.action_id)))