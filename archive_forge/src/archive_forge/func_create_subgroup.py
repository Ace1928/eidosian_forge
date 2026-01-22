from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def create_subgroup(self, parents, grouprep, realm='master'):
    """ Create a Keycloak subgroup.

        :param parents: list of one or more parent groups
        :param grouprep: a GroupRepresentation of the group to be created. Must contain at minimum the field name.
        :return: HTTPResponse object on success
        """
    parent_id = '---UNDETERMINED---'
    try:
        parent_id = self.get_subgroup_direct_parent(parents, realm)
        if not parent_id:
            raise Exception('Could not determine subgroup parent ID for given parent chain {0}. Assure that all parents exist already and the list is complete and properly ordered, starts with an ID or starts at the top level'.format(parents))
        parent_id = parent_id['id']
        url = URL_GROUP_CHILDREN.format(url=self.baseurl, realm=realm, groupid=parent_id)
        return open_url(url, method='POST', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(grouprep), validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not create subgroup %s for parent group %s in realm %s: %s' % (grouprep['name'], parent_id, realm, str(e)))