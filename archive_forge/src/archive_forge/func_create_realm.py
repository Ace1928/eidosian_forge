from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def create_realm(self, realmrep):
    """ Create a realm in keycloak
        :param realmrep: Realm representation of realm to be created.
        :return: HTTPResponse object on success
        """
    realm_url = URL_REALMS.format(url=self.baseurl)
    try:
        return open_url(realm_url, method='POST', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(realmrep), validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not create realm %s: %s' % (realmrep['id'], str(e)), exception=traceback.format_exc())