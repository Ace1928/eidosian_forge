from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def update_authz_authorization_scope(self, payload, id, client_id, realm):
    """Update an authorization scope for a Keycloak client"""
    url = URL_AUTHZ_AUTHORIZATION_SCOPE.format(url=self.baseurl, id=id, client_id=client_id, realm=realm)
    try:
        return open_url(url, method='PUT', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(payload), validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not create update scope %s for client %s in realm %s: %s' % (payload['name'], client_id, realm, str(e)))