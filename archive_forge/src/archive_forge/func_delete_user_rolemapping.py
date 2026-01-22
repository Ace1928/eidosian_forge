from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def delete_user_rolemapping(self, uid, cid, role_rep, realm='master'):
    """ Delete the rolemapping of a client in a specified user on the Keycloak server.

        :param uid: ID of the user from which to remove the rolemappings.
        :param cid: ID of the client from which to remove the rolemappings.
        :param role_rep: Representation of the role to remove from rolemappings.
        :param realm: Realm from which to remove the rolemappings.
        :return: None.
        """
    if cid is None:
        user_realm_rolemappings_url = URL_REALM_ROLEMAPPINGS.format(url=self.baseurl, realm=realm, id=uid)
        try:
            open_url(user_realm_rolemappings_url, method='DELETE', http_agent=self.http_agent, headers=self.restheaders, data=json.dumps(role_rep), validate_certs=self.validate_certs, timeout=self.connection_timeout)
        except Exception as e:
            self.fail_open_url(e, msg='Could not remove roles %s from userId %s, realm %s: %s' % (json.dumps(role_rep), uid, realm, str(e)))
    else:
        user_client_rolemappings_url = URL_CLIENT_USER_ROLEMAPPINGS.format(url=self.baseurl, realm=realm, id=uid, client=cid)
        try:
            open_url(user_client_rolemappings_url, method='DELETE', http_agent=self.http_agent, headers=self.restheaders, data=json.dumps(role_rep), validate_certs=self.validate_certs, timeout=self.connection_timeout)
        except Exception as e:
            self.fail_open_url(e, msg='Could not remove roles %s for client %s from userId %s, realm %s: %s' % (json.dumps(role_rep), cid, uid, realm, str(e)))