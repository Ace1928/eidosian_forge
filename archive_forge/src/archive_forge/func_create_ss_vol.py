from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
def create_ss_vol(self):
    post_data = dict(snapshotImageId=self.snapshot_image_id, fullThreshold=self.full_threshold, name=self.name, viewMode=self.view_mode, repositoryPercentage=self.repo_percentage, repositoryPoolId=self.pool_id)
    rc, create_resp = request(self.url + 'storage-systems/%s/snapshot-volumes' % self.ssid, data=json.dumps(post_data), headers=HEADERS, url_username=self.user, url_password=self.pwd, validate_certs=self.certs, method='POST')
    self.ss_vol = create_resp
    if self.ss_vol_needs_update:
        self.update_ss_vol()
    else:
        self.module.exit_json(changed=True, **create_resp)