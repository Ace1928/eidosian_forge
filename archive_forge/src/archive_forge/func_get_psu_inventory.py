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
def get_psu_inventory(self):
    result = {}
    psu_list = []
    psu_results = []
    key = 'PowerSupplies'
    properties = ['Name', 'Model', 'SerialNumber', 'PartNumber', 'Manufacturer', 'FirmwareVersion', 'PowerCapacityWatts', 'PowerSupplyType', 'Status']
    for chassis_uri in self.chassis_uris:
        response = self.get_request(self.root_uri + chassis_uri)
        if response['ret'] is False:
            return response
        result['ret'] = True
        data = response['data']
        if 'Power' in data:
            power_uri = data[u'Power'][u'@odata.id']
        else:
            continue
        response = self.get_request(self.root_uri + power_uri)
        data = response['data']
        if key not in data:
            return {'ret': False, 'msg': 'Key %s not found' % key}
        psu_list = data[key]
        for psu in psu_list:
            psu_not_present = False
            psu_data = {}
            for property in properties:
                if property in psu:
                    if psu[property] is not None:
                        if property == 'Status':
                            if 'State' in psu[property]:
                                if psu[property]['State'] == 'Absent':
                                    psu_not_present = True
                        psu_data[property] = psu[property]
            if psu_not_present:
                continue
            psu_results.append(psu_data)
    result['entries'] = psu_results
    if not result['entries']:
        return {'ret': False, 'msg': 'No PowerSupply objects found'}
    return result