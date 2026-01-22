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
def get_nic_inventory(self, resource_uri):
    result = {}
    nic_list = []
    nic_results = []
    key = 'EthernetInterfaces'
    response = self.get_request(self.root_uri + resource_uri)
    if response['ret'] is False:
        return response
    result['ret'] = True
    data = response['data']
    if key not in data:
        return {'ret': False, 'msg': 'Key %s not found' % key}
    ethernetinterfaces_uri = data[key]['@odata.id']
    response = self.get_request(self.root_uri + ethernetinterfaces_uri)
    if response['ret'] is False:
        return response
    result['ret'] = True
    data = response['data']
    for nic in data[u'Members']:
        nic_list.append(nic[u'@odata.id'])
    for n in nic_list:
        nic = self.get_nic(n)
        if nic['ret']:
            nic_results.append(nic['entries'])
    result['entries'] = nic_results
    return result