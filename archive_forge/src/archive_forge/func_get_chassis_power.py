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
def get_chassis_power(self):
    result = {}
    key = 'Power'
    properties = ['Name', 'PowerAllocatedWatts', 'PowerAvailableWatts', 'PowerCapacityWatts', 'PowerConsumedWatts', 'PowerMetrics', 'PowerRequestedWatts', 'RelatedItem', 'Status']
    chassis_power_results = []
    for chassis_uri in self.chassis_uris:
        chassis_power_result = {}
        response = self.get_request(self.root_uri + chassis_uri)
        if response['ret'] is False:
            return response
        result['ret'] = True
        data = response['data']
        if key in data:
            response = self.get_request(self.root_uri + data[key]['@odata.id'])
            data = response['data']
            if 'PowerControl' in data:
                if len(data['PowerControl']) > 0:
                    data = data['PowerControl'][0]
                    for property in properties:
                        if property in data:
                            chassis_power_result[property] = data[property]
            chassis_power_results.append(chassis_power_result)
    if len(chassis_power_results) > 0:
        result['entries'] = chassis_power_results
        return result
    else:
        return {'ret': False, 'msg': 'Power information not found.'}