from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_client_roles_by_id_composite_rolemappings(self, rid, cid, realm='master'):
    """ Fetch a role by its id on the Keycloak server.

        :param rid: ID of the composite role.
        :param cid: ID of the client from which to obtain the rolemappings.
        :param realm: Realm from which to obtain the rolemappings.
        :return: The role.
        """
    client_roles_url = URL_ROLES_BY_ID_COMPOSITES_CLIENTS.format(url=self.baseurl, realm=realm, id=rid, cid=cid)
    try:
        return json.loads(to_native(open_url(client_roles_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except Exception as e:
        self.fail_open_url(e, msg='Could not fetch role for id %s and cid %s in realm %s: %s' % (rid, cid, realm, str(e)))