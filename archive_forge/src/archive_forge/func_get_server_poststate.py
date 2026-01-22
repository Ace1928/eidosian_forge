from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
import time
def get_server_poststate(self):
    response = self.get_request(self.root_uri + self.systems_uri)
    if not response['ret']:
        return response
    server_data = response['data']
    if 'Hpe' in server_data['Oem']:
        return {'ret': True, 'server_poststate': server_data['Oem']['Hpe']['PostState']}
    else:
        return {'ret': True, 'server_poststate': server_data['Oem']['Hp']['PostState']}