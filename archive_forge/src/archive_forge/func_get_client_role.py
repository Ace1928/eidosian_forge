from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_client_role(self, name, clientid, realm='master'):
    """ Fetch a keycloak client role from the provided realm using the role's name.

        :param name: Name of the role to fetch.
        :param clientid: Client id for the client role
        :param realm: Realm in which the role resides
        :return: Dict of role representation
        If the role does not exist, None is returned.
        """
    cid = self.get_client_id(clientid, realm=realm)
    if cid is None:
        self.module.fail_json(msg='Could not find client %s in realm %s' % (clientid, realm))
    role_url = URL_CLIENT_ROLE.format(url=self.baseurl, realm=realm, id=cid, name=quote(name, safe=''))
    try:
        return json.loads(to_native(open_url(role_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except HTTPError as e:
        if e.code == 404:
            return None
        else:
            self.fail_open_url(e, msg='Could not fetch role %s in client %s of realm %s: %s' % (name, clientid, realm, str(e)))
    except Exception as e:
        self.module.fail_json(msg='Could not fetch role %s for client %s in realm %s: %s' % (name, clientid, realm, str(e)))