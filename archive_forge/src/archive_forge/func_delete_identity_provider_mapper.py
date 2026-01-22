from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def delete_identity_provider_mapper(self, mid, alias, realm='master'):
    """ Delete an identity provider.
        :param mid: Unique ID of the mapper to delete.
        :param alias: Alias of the identity provider.
        :param realm: Realm in which this identity provider resides, default "master".
        """
    mapper_url = URL_IDENTITY_PROVIDER_MAPPER.format(url=self.baseurl, realm=realm, alias=alias, id=mid)
    try:
        return open_url(mapper_url, method='DELETE', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Unable to delete mapper %s for identity provider %s in realm %s: %s' % (mid, alias, realm, str(e)))