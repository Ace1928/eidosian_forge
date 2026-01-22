from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def create_authz_authorization_scope(self, payload, client_id, realm):
    """Create an authorization scope for a Keycloak client"""
    url = URL_AUTHZ_AUTHORIZATION_SCOPES.format(url=self.baseurl, client_id=client_id, realm=realm)
    try:
        return open_url(url, method='POST', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(payload), validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not create authorization scope %s for client %s in realm %s: %s' % (payload['name'], client_id, realm, str(e)))