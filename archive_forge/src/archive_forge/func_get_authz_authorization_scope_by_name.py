from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_authz_authorization_scope_by_name(self, name, client_id, realm):
    url = URL_AUTHZ_AUTHORIZATION_SCOPES.format(url=self.baseurl, client_id=client_id, realm=realm)
    search_url = '%s/search?name=%s' % (url, quote(name, safe=''))
    try:
        return json.loads(to_native(open_url(search_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except Exception:
        return False