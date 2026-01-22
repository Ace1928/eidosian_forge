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
def get_boot_order(self, systems_uri):
    result = {}
    response = self.get_request(self.root_uri + systems_uri)
    if response['ret'] is False:
        return response
    result['ret'] = True
    data = response['data']
    if 'Boot' not in data or 'BootOrder' not in data['Boot']:
        return {'ret': False, 'msg': 'Key BootOrder not found'}
    boot = data['Boot']
    boot_order = boot['BootOrder']
    boot_options_dict = self._get_boot_options_dict(boot)
    boot_device_list = []
    for ref in boot_order:
        boot_device_list.append(boot_options_dict.get(ref, {'BootOptionReference': ref}))
    result['entries'] = boot_device_list
    return result