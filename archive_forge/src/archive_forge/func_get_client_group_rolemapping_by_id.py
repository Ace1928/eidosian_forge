from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_client_group_rolemapping_by_id(self, gid, cid, rid, realm='master'):
    """ Obtain client representation by id

        :param gid: ID of the group from which to obtain the rolemappings.
        :param cid: ID of the client from which to obtain the rolemappings.
        :param rid: ID of the role.
        :param realm: client from this realm
        :return: dict of rolemapping representation or None if none matching exist
        """
    rolemappings_url = URL_CLIENT_GROUP_ROLEMAPPINGS.format(url=self.baseurl, realm=realm, id=gid, client=cid)
    try:
        rolemappings = json.loads(to_native(open_url(rolemappings_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
        for role in rolemappings:
            if rid == role['id']:
                return role
    except Exception as e:
        self.fail_open_url(e, msg='Could not fetch rolemappings for client %s in group %s, realm %s: %s' % (cid, gid, realm, str(e)))
    return None