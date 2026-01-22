from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
def remove_ss_vol(self):
    rc, resp = request(self.url + 'storage-systems/%s/snapshot-volumes/%s' % (self.ssid, self.ss_vol['id']), headers=HEADERS, url_username=self.user, url_password=self.pwd, validate_certs=self.certs, method='DELETE')
    self.module.exit_json(changed=True, msg='Volume successfully deleted')