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
def get_memory_inventory(self, systems_uri):
    result = {}
    memory_list = []
    memory_results = []
    key = 'Memory'
    properties = ['Id', 'SerialNumber', 'MemoryDeviceType', 'PartNumber', 'MemoryLocation', 'RankCount', 'CapacityMiB', 'OperatingMemoryModes', 'Status', 'Manufacturer', 'Name']
    response = self.get_request(self.root_uri + systems_uri)
    if response['ret'] is False:
        return response
    result['ret'] = True
    data = response['data']
    if key not in data:
        return {'ret': False, 'msg': 'Key %s not found' % key}
    memory_uri = data[key]['@odata.id']
    response = self.get_request(self.root_uri + memory_uri)
    if response['ret'] is False:
        return response
    result['ret'] = True
    data = response['data']
    for dimm in data[u'Members']:
        memory_list.append(dimm[u'@odata.id'])
    for m in memory_list:
        dimm = {}
        uri = self.root_uri + m
        response = self.get_request(uri)
        if response['ret'] is False:
            return response
        data = response['data']
        if 'Status' in data:
            if 'State' in data['Status']:
                if data['Status']['State'] == 'Absent':
                    continue
        else:
            continue
        for property in properties:
            if property in data:
                dimm[property] = data[property]
        memory_results.append(dimm)
    result['entries'] = memory_results
    return result