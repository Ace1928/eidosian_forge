from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_identity_provider_mappers(self, alias, realm='master'):
    """ Fetch representations for identity provider mappers
        :param alias: Alias of the identity provider.
        :param realm: realm to be queried
        :return: list of representations for identity provider mappers
        """
    mappers_url = URL_IDENTITY_PROVIDER_MAPPERS.format(url=self.baseurl, realm=realm, alias=alias)
    try:
        return json.loads(to_native(open_url(mappers_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except ValueError as e:
        self.module.fail_json(msg='API returned incorrect JSON when trying to obtain list of identity provider mappers for idp %s in realm %s: %s' % (alias, realm, str(e)))
    except Exception as e:
        self.fail_open_url(e, msg='Could not obtain list of identity provider mappers for idp %s in realm %s: %s' % (alias, realm, str(e)))