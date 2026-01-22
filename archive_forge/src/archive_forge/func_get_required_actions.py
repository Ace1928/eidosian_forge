from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_required_actions(self, realm='master'):
    """
        Get required actions.
        :param realm: Realm name (not id).
        :return:      List of representations of the required actions.
        """
    try:
        required_actions = json.load(open_url(URL_AUTHENTICATION_REQUIRED_ACTIONS.format(url=self.baseurl, realm=realm), method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs))
        return required_actions
    except Exception:
        return None