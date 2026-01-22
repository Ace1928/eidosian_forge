from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
def update_ss_vol(self):
    post_data = dict(fullThreshold=self.full_threshold)
    rc, resp = request(self.url + 'storage-systems/%s/snapshot-volumes/%s' % (self.ssid, self.ss_vol['id']), data=json.dumps(post_data), headers=HEADERS, url_username=self.user, url_password=self.pwd, method='POST', validate_certs=self.certs)
    self.module.exit_json(changed=True, **resp)