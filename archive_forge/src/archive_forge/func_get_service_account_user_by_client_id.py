from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_service_account_user_by_client_id(self, client_id, realm='master'):
    """ Fetch a keycloak service account user within a realm based on its client_id.

        If the user does not exist, None is returned.
        :param client_id: clientId of the service account user to fetch.
        :param realm: Realm in which the user resides; default 'master'
        """
    cid = self.get_client_id(client_id, realm=realm)
    service_account_user_url = URL_CLIENT_SERVICE_ACCOUNT_USER.format(url=self.baseurl, realm=realm, id=cid)
    try:
        return json.loads(to_native(open_url(service_account_user_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except ValueError as e:
        self.module.fail_json(msg='API returned incorrect JSON when trying to obtain the service-account-user for realm %s and client_id %s: %s' % (realm, client_id, str(e)))
    except Exception as e:
        self.fail_open_url(e, msg='Could not obtain the service-account-user for realm %s and client_id %s: %s' % (realm, client_id, str(e)))