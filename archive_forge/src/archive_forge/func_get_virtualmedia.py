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
def get_virtualmedia(self, resource_uri):
    result = {}
    virtualmedia_list = []
    virtualmedia_results = []
    key = 'VirtualMedia'
    properties = ['Description', 'ConnectedVia', 'Id', 'MediaTypes', 'Image', 'ImageName', 'Name', 'WriteProtected', 'TransferMethod', 'TransferProtocolType']
    response = self.get_request(self.root_uri + resource_uri)
    if response['ret'] is False:
        return response
    result['ret'] = True
    data = response['data']
    if key not in data:
        return {'ret': False, 'msg': 'Key %s not found' % key}
    virtualmedia_uri = data[key]['@odata.id']
    response = self.get_request(self.root_uri + virtualmedia_uri)
    if response['ret'] is False:
        return response
    result['ret'] = True
    data = response['data']
    for virtualmedia in data[u'Members']:
        virtualmedia_list.append(virtualmedia[u'@odata.id'])
    for n in virtualmedia_list:
        virtualmedia = {}
        uri = self.root_uri + n
        response = self.get_request(uri)
        if response['ret'] is False:
            return response
        data = response['data']
        for property in properties:
            if property in data:
                virtualmedia[property] = data[property]
        virtualmedia_results.append(virtualmedia)
    result['entries'] = virtualmedia_results
    return result