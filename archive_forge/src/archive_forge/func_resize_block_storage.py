from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def resize_block_storage(self, volume_name, region, desired_size):
    if not desired_size:
        return False
    volume = self.get_block_storage_by_name(volume_name, region)
    if volume['size_gigabytes'] == desired_size:
        return False
    data = {'type': 'resize', 'size_gigabytes': desired_size}
    resp = self.rest.post('volumes/{0}/actions'.format(volume['id']), data=data)
    if resp.status_code == 202:
        return self.poll_action_for_complete_status(resp.json['action']['id'])
    else:
        raise DOBlockStorageException(resp.json['message'])