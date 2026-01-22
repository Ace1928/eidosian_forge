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
def check_location_uri(self, resp_data, resp_uri):
    vendor = self._get_vendor()['Vendor']
    rsp_uri = ''
    for loc in resp_data['Location']:
        if loc['Language'] == 'en':
            rsp_uri = loc['Uri']
            if vendor == 'HPE':
                if isinstance(loc['Uri'], dict) and 'extref' in loc['Uri'].keys():
                    rsp_uri = loc['Uri']['extref']
    if not rsp_uri:
        msg = "Language 'en' not found in BIOS Attribute Registries location, URI: %s, response: %s"
        return {'ret': False, 'msg': msg % (resp_uri, str(resp_data))}
    res = self.get_request(self.root_uri + rsp_uri)
    if res['ret'] is False:
        if vendor == 'HPE':
            override_headers = {'Accept-Encoding': 'gzip'}
            res = self.get_request(self.root_uri + rsp_uri, override_headers=override_headers)
    if res['ret']:
        return {'ret': True, 'rsp_data': res['data'], 'rsp_uri': rsp_uri}
    return res