from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def delete_component(self, cid, realm='master'):
    """ Delete an component.
        :param cid: Unique ID of the component.
        :param realm: Realm in which this component resides, default "master".
        """
    comp_url = URL_COMPONENT.format(url=self.baseurl, realm=realm, id=cid)
    try:
        return open_url(comp_url, method='DELETE', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Unable to delete component %s in realm %s: %s' % (cid, realm, str(e)))