from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def create_realm_role(self, rolerep, realm='master'):
    """ Create a Keycloak realm role.

        :param rolerep: a RoleRepresentation of the role to be created. Must contain at minimum the field name.
        :return: HTTPResponse object on success
        """
    roles_url = URL_REALM_ROLES.format(url=self.baseurl, realm=realm)
    try:
        if 'composites' in rolerep:
            keycloak_compatible_composites = self.convert_role_composites(rolerep['composites'])
            rolerep['composites'] = keycloak_compatible_composites
        return open_url(roles_url, method='POST', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(rolerep), validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not create role %s in realm %s: %s' % (rolerep['name'], realm, str(e)))