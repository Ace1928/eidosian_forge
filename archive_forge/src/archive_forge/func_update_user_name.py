from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def update_user_name(self, user):
    if not user.get('account_updatename'):
        return {'ret': False, 'msg': 'Must provide account_updatename for UpdateUserName command'}
    response = self._find_account_uri(username=user.get('account_username'), acct_id=user.get('account_id'))
    if not response['ret']:
        return response
    uri = response['uri']
    payload = {'UserName': user['account_updatename']}
    return self.patch_request(self.root_uri + uri, payload, check_pyld=True)