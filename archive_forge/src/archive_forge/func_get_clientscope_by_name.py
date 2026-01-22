from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_clientscope_by_name(self, name, realm='master'):
    """ Fetch a keycloak clientscope within a realm based on its name.

        The Keycloak API does not allow filtering of the clientscopes resource by name.
        As a result, this method first retrieves the entire list of clientscopes - name and ID -
        then performs a second query to fetch the group.

        If the clientscope does not exist, None is returned.
        :param name: Name of the clientscope to fetch.
        :param realm: Realm in which the clientscope resides; default 'master'
        """
    try:
        all_clientscopes = self.get_clientscopes(realm=realm)
        for clientscope in all_clientscopes:
            if clientscope['name'] == name:
                return self.get_clientscope_by_clientscopeid(clientscope['id'], realm=realm)
        return None
    except Exception as e:
        self.module.fail_json(msg='Could not fetch clientscope %s in realm %s: %s' % (name, realm, str(e)))