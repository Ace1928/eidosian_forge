from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_user_groups(self, user_id, realm='master'):
    """
        Get groups for a user.
        :param user_id: User ID
        :param realm: Realm
        :return: Representation of the client groups.
        """
    try:
        groups = []
        user_groups_url = URL_USER_GROUPS.format(url=self.baseurl, realm=realm, id=user_id)
        user_groups = json.load(open_url(user_groups_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs))
        for user_group in user_groups:
            groups.append(user_group['name'])
        return groups
    except Exception as e:
        self.fail_open_url(e, msg='Could not get groups for user %s in realm %s: %s' % (user_id, realm, str(e)))