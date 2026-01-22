from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_clients(self, realm='master', filter=None):
    """ Obtains client representations for clients in a realm

        :param realm: realm to be queried
        :param filter: if defined, only the client with clientId specified in the filter is returned
        :return: list of dicts of client representations
        """
    clientlist_url = URL_CLIENTS.format(url=self.baseurl, realm=realm)
    if filter is not None:
        clientlist_url += '?clientId=%s' % filter
    try:
        return json.loads(to_native(open_url(clientlist_url, http_agent=self.http_agent, method='GET', headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except ValueError as e:
        self.module.fail_json(msg='API returned incorrect JSON when trying to obtain list of clients for realm %s: %s' % (realm, str(e)))
    except Exception as e:
        self.fail_open_url(e, msg='Could not obtain list of clients for realm %s: %s' % (realm, str(e)))