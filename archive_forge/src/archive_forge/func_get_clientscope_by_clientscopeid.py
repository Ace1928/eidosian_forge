from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_clientscope_by_clientscopeid(self, cid, realm='master'):
    """ Fetch a keycloak clientscope from the provided realm using the clientscope's unique ID.

        If the clientscope does not exist, None is returned.

        gid is a UUID provided by the Keycloak API
        :param cid: UUID of the clientscope to be returned
        :param realm: Realm in which the clientscope resides; default 'master'.
        """
    clientscope_url = URL_CLIENTSCOPE.format(url=self.baseurl, realm=realm, id=cid)
    try:
        return json.loads(to_native(open_url(clientscope_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except HTTPError as e:
        if e.code == 404:
            return None
        else:
            self.fail_open_url(e, msg='Could not fetch clientscope %s in realm %s: %s' % (cid, realm, str(e)))
    except Exception as e:
        self.module.fail_json(msg='Could not clientscope group %s in realm %s: %s' % (cid, realm, str(e)))