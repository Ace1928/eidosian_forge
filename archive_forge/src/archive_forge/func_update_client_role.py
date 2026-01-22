from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def update_client_role(self, rolerep, clientid, realm='master'):
    """ Update an existing client role.

        :param rolerep: A RoleRepresentation of the updated role.
        :param clientid: Client id for the client role
        :param realm: Realm in which the role resides
        :return HTTPResponse object on success
        """
    cid = self.get_client_id(clientid, realm=realm)
    if cid is None:
        self.module.fail_json(msg='Could not find client %s in realm %s' % (clientid, realm))
    role_url = URL_CLIENT_ROLE.format(url=self.baseurl, realm=realm, id=cid, name=quote(rolerep['name'], safe=''))
    try:
        composites = None
        if 'composites' in rolerep:
            composites = copy.deepcopy(rolerep['composites'])
            del rolerep['composites']
        update_role_response = open_url(role_url, method='PUT', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(rolerep), validate_certs=self.validate_certs)
        if composites is not None:
            self.update_role_composites(rolerep=rolerep, clientid=clientid, composites=composites, realm=realm)
        return update_role_response
    except Exception as e:
        self.fail_open_url(e, msg='Could not update role %s for client %s in realm %s: %s' % (rolerep['name'], clientid, realm, str(e)))