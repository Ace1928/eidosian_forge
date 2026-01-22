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
def virtual_media_eject(self, options, resource_type='Manager'):
    image_url = options.get('image_url')
    if not image_url:
        return {'ret': False, 'msg': 'image_url option required for VirtualMediaEject'}
    if resource_type == 'Systems':
        resource_uri = self.systems_uri
    elif resource_type == 'Manager':
        resource_uri = self.manager_uri
    response = self.get_request(self.root_uri + resource_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    if 'VirtualMedia' not in data:
        return {'ret': False, 'msg': 'VirtualMedia resource not found'}
    virt_media_uri = data['VirtualMedia']['@odata.id']
    response = self.get_request(self.root_uri + virt_media_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    virt_media_list = []
    for member in data[u'Members']:
        virt_media_list.append(member[u'@odata.id'])
    resources, headers = self._read_virt_media_resources(virt_media_list)
    uri, data, eject = self._find_virt_media_to_eject(resources, image_url)
    if uri and eject:
        if 'Actions' not in data or '#VirtualMedia.EjectMedia' not in data['Actions']:
            h = headers[uri]
            if 'allow' in h:
                methods = [m.strip() for m in h.get('allow').split(',')]
                if 'PATCH' not in methods:
                    return {'ret': False, 'msg': '%s action not found and PATCH not allowed' % '#VirtualMedia.EjectMedia'}
            return self.virtual_media_eject_via_patch(uri)
        else:
            action = data['Actions']['#VirtualMedia.EjectMedia']
            if 'target' not in action:
                return {'ret': False, 'msg': 'target URI property missing from Action #VirtualMedia.EjectMedia'}
            action_uri = action['target']
            payload = {}
            response = self.post_request(self.root_uri + action_uri, payload)
            if response['ret'] is False:
                return response
            return {'ret': True, 'changed': True, 'msg': 'VirtualMedia ejected'}
    elif uri and (not eject):
        return {'ret': True, 'changed': False, 'msg': "VirtualMedia image '%s' already ejected" % image_url}
    else:
        return {'ret': False, 'changed': False, 'msg': "No VirtualMedia resource found with image '%s' inserted" % image_url}