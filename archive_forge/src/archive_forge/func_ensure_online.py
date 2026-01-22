from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def ensure_online(self, database_id):
    end_time = time.monotonic() + self.wait_timeout
    while time.monotonic() < end_time:
        response = self.rest.get('databases/{0}'.format(database_id))
        json_data = response.json
        database = json_data.get('database', None)
        if database is not None:
            status = database.get('status', None)
            if status is not None:
                if status == 'online':
                    return json_data
        time.sleep(10)
    self.module.fail_json(msg='Waiting for database online timeout')