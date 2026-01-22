from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_realm_user_composite_rolemappings(self, uid, realm='master'):
    """ Fetch the composite role of a realm for a specified user on the Keycloak server.

        :param uid: ID of the user from which to obtain the rolemappings.
        :param realm: Realm from which to obtain the rolemappings.
        :return: The effective rollemappings of specified client and user of the realm (default "master").
        """
    composite_rolemappings_url = URL_REALM_ROLEMAPPINGS_COMPOSITE.format(url=self.baseurl, realm=realm, id=uid)
    try:
        return json.loads(to_native(open_url(composite_rolemappings_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except Exception as e:
        self.fail_open_url(e, msg='Could not fetch effective rolemappings for user %s, realm %s: %s' % (uid, realm, str(e)))