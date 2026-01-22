from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
def create_snapshot_group(self):
    self.post_data = dict(baseMappableObjectId=self.volume_id, name=self.name, repositoryPercentage=self.repo_pct, warningThreshold=self.warning_threshold, autoDeleteLimit=self.delete_limit, fullPolicy=self.full_policy, storagePoolId=self.pool_id)
    snapshot = 'storage-systems/%s/snapshot-groups' % self.ssid
    url = self.url + snapshot
    try:
        rc, self.ssg_data = request(url, data=json.dumps(self.post_data), method='POST', headers=HEADERS, url_username=self.user, url_password=self.pwd, validate_certs=self.certs)
    except Exception as err:
        self.module.fail_json(msg='Failed to create snapshot group. ' + 'Snapshot group [%s]. Id [%s]. Error [%s].' % (self.name, self.ssid, to_native(err)))
    if not self.snapshot_group_id:
        self.snapshot_group_id = self.ssg_data['id']
    if self.ssg_needs_update:
        self.update_ssg()
    else:
        self.module.exit_json(changed=True, **self.ssg_data)