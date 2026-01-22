from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib_parse import urlparse, parse_qs, urlencode
from urllib.parse import urljoin
from base64 import urlsafe_b64encode
import hashlib
def openshift_logout(self):
    name = get_oauthaccesstoken_objectname_from_token(self.auth_api_key)
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': 'Bearer {0}'.format(self.auth_api_key)}
    url = '{0}/apis/oauth.openshift.io/v1/useroauthaccesstokens/{1}'.format(self.con_host, name)
    json = {'apiVersion': 'oauth.openshift.io/v1', 'kind': 'DeleteOptions', 'gracePeriodSeconds': 0}
    ret = requests.delete(url, json=json, verify=self.con_verify_ca, headers=headers)
    if ret.status_code != 200:
        self.fail_json(msg="Couldn't delete user oauth access token '{0}' due to: {1}".format(name, ret.json().get('message')), status_code=ret.status_code)
    return True