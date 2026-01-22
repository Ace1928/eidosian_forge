from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def wait_status(self, droplet_id, desired_statuses):
    end_time = time.monotonic() + self.wait_timeout
    while time.monotonic() < end_time:
        response = self.rest.get('droplets/{0}'.format(droplet_id))
        json_data = response.json
        status_code = response.status_code
        message = json_data.get('message', 'no error message')
        droplet = json_data.get('droplet', None)
        droplet_status = droplet.get('status', None) if droplet else None
        if droplet is None or droplet_status is None:
            self.module.fail_json(changed=False, msg=DODroplet.failure_message['unexpected'].format('no Droplet or status'))
        if status_code >= 400:
            self.module.fail_json(changed=False, msg=DODroplet.failure_message['failed_to'].format('get', 'Droplet', status_code, message))
        if droplet_status in desired_statuses:
            return
        time.sleep(self.sleep_interval)
    self.module.fail_json(msg='Wait for Droplet [{0}] status timeout'.format(','.join(desired_statuses)))