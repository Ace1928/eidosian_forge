from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def remove_authz_custom_policy(self, policy_id, client_id, realm):
    """Remove a custom policy from a Keycloak client"""
    url = URL_AUTHZ_CUSTOM_POLICIES.format(url=self.baseurl, client_id=client_id, realm=realm)
    delete_url = '%s/%s' % (url, policy_id)
    try:
        return open_url(delete_url, method='DELETE', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not delete custom policy %s for client %s in realm %s: %s' % (id, client_id, realm, str(e)))