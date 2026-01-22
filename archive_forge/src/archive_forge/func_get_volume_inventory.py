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
def get_volume_inventory(self, systems_uri):
    result = {'entries': []}
    controller_list = []
    volume_list = []
    properties = ['Id', 'Name', 'RAIDType', 'VolumeType', 'BlockSizeBytes', 'Capacity', 'CapacityBytes', 'CapacitySources', 'Encrypted', 'EncryptionTypes', 'Identifiers', 'Operations', 'OptimumIOSizeBytes', 'AccessCapabilities', 'AllocatedPools', 'Status']
    response = self.get_request(self.root_uri + systems_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    if 'SimpleStorage' not in data and 'Storage' not in data:
        return {'ret': False, 'msg': 'SimpleStorage and Storage resource                      not found'}
    if 'Storage' in data:
        storage_uri = data[u'Storage'][u'@odata.id']
        response = self.get_request(self.root_uri + storage_uri)
        if response['ret'] is False:
            return response
        result['ret'] = True
        data = response['data']
        if data.get('Members'):
            for controller in data[u'Members']:
                controller_list.append(controller[u'@odata.id'])
            for idx, c in enumerate(controller_list):
                uri = self.root_uri + c
                response = self.get_request(uri)
                if response['ret'] is False:
                    return response
                data = response['data']
                controller_name = 'Controller %s' % str(idx)
                if 'Controllers' in data:
                    response = self.get_request(self.root_uri + data['Controllers'][u'@odata.id'])
                    if response['ret'] is False:
                        return response
                    c_data = response['data']
                    if c_data.get('Members') and c_data['Members']:
                        response = self.get_request(self.root_uri + c_data['Members'][0][u'@odata.id'])
                        if response['ret'] is False:
                            return response
                        member_data = response['data']
                        if member_data:
                            if 'Name' in member_data:
                                controller_name = member_data['Name']
                            else:
                                controller_id = member_data.get('Id', '1')
                                controller_name = 'Controller %s' % controller_id
                elif 'StorageControllers' in data:
                    sc = data['StorageControllers']
                    if sc:
                        if 'Name' in sc[0]:
                            controller_name = sc[0]['Name']
                        else:
                            sc_id = sc[0].get('Id', '1')
                            controller_name = 'Controller %s' % sc_id
                volume_results = []
                volume_list = []
                if 'Volumes' in data:
                    volumes_uri = data[u'Volumes'][u'@odata.id']
                    response = self.get_request(self.root_uri + volumes_uri)
                    data = response['data']
                    if data.get('Members'):
                        for volume in data[u'Members']:
                            volume_list.append(volume[u'@odata.id'])
                        for v in volume_list:
                            uri = self.root_uri + v
                            response = self.get_request(uri)
                            if response['ret'] is False:
                                return response
                            data = response['data']
                            volume_result = {}
                            for property in properties:
                                if property in data:
                                    if data[property] is not None:
                                        volume_result[property] = data[property]
                            drive_id_list = []
                            if 'Links' in data:
                                if 'Drives' in data[u'Links']:
                                    for link in data[u'Links'][u'Drives']:
                                        drive_id_link = link[u'@odata.id']
                                        drive_id = drive_id_link.split('/')[-1]
                                        drive_id_list.append({'Id': drive_id})
                                    volume_result['Linked_drives'] = drive_id_list
                            volume_results.append(volume_result)
                volumes = {'Controller': controller_name, 'Volumes': volume_results}
                result['entries'].append(volumes)
    else:
        return {'ret': False, 'msg': 'Storage resource not found'}
    return result