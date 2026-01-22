from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def update_realm_role(self, rolerep, realm='master'):
    """ Update an existing realm role.

        :param rolerep: A RoleRepresentation of the updated role.
        :return HTTPResponse object on success
        """
    role_url = URL_REALM_ROLE.format(url=self.baseurl, realm=realm, name=quote(rolerep['name']), safe='')
    try:
        composites = None
        if 'composites' in rolerep:
            composites = copy.deepcopy(rolerep['composites'])
            del rolerep['composites']
        role_response = open_url(role_url, method='PUT', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(rolerep), validate_certs=self.validate_certs)
        if composites is not None:
            self.update_role_composites(rolerep=rolerep, composites=composites, realm=realm)
        return role_response
    except Exception as e:
        self.fail_open_url(e, msg='Could not update role %s in realm %s: %s' % (rolerep['name'], realm, str(e)))