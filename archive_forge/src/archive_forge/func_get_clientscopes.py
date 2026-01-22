from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_clientscopes(self, realm='master'):
    """ Fetch the name and ID of all clientscopes on the Keycloak server.

        To fetch the full data of the group, make a subsequent call to
        get_clientscope_by_clientscopeid, passing in the ID of the group you wish to return.

        :param realm: Realm in which the clientscope resides; default 'master'.
        :return The clientscopes of this realm (default "master")
        """
    clientscopes_url = URL_CLIENTSCOPES.format(url=self.baseurl, realm=realm)
    try:
        return json.loads(to_native(open_url(clientscopes_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except Exception as e:
        self.fail_open_url(e, msg='Could not fetch list of clientscopes in realm %s: %s' % (realm, str(e)))