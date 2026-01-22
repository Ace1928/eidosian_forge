from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def delete_block_storage(self):
    volume_name = self.get_key_or_fail('volume_name')
    region = self.get_key_or_fail('region')
    url = 'volumes?name={0}&region={1}'.format(volume_name, region)
    attached_droplet_id = self.get_attached_droplet_ID(volume_name, region)
    if attached_droplet_id is not None:
        self.attach_detach_block_storage('detach', volume_name, region, attached_droplet_id)
    response = self.rest.delete(url)
    status = response.status_code
    json = response.json
    if status == 204:
        self.module.exit_json(changed=True)
    elif status == 404:
        self.module.exit_json(changed=False)
    else:
        raise DOBlockStorageException(json['message'])