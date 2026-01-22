from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
def update_ssg(self):
    self.post_data = dict(warningThreshold=self.warning_threshold, autoDeleteLimit=self.delete_limit, fullPolicy=self.full_policy, rollbackPriority=self.rollback_priority)
    url = self.url + 'storage-systems/%s/snapshot-groups/%s' % (self.ssid, self.snapshot_group_id)
    try:
        rc, self.ssg_data = request(url, data=json.dumps(self.post_data), method='POST', headers=HEADERS, url_username=self.user, url_password=self.pwd, validate_certs=self.certs)
    except Exception as err:
        self.module.fail_json(msg='Failed to update snapshot group. ' + 'Snapshot group [%s]. Id [%s]. Error [%s].' % (self.name, self.ssid, to_native(err)))