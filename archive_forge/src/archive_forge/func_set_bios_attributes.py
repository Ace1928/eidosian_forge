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
def set_bios_attributes(self, attributes):
    response = self.get_request(self.root_uri + self.systems_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    bios_uri = data.get('Bios', {}).get('@odata.id')
    if bios_uri is None:
        return {'ret': False, 'msg': 'Bios resource not found'}
    response = self.get_request(self.root_uri + bios_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    attrs_to_patch = dict(attributes)
    attrs_bad = {}
    for attr_name, attr_value in attributes.items():
        if attr_name not in data[u'Attributes']:
            attrs_bad.update({attr_name: attr_value})
            del attrs_to_patch[attr_name]
            continue
        if data[u'Attributes'][attr_name] == attributes[attr_name]:
            del attrs_to_patch[attr_name]
    warning = ''
    if attrs_bad:
        warning = 'Unsupported attributes %s' % attrs_bad
    if not attrs_to_patch:
        return {'ret': True, 'changed': False, 'msg': 'BIOS attributes already set', 'warning': warning}
    set_bios_attr_uri = data.get('@Redfish.Settings', {}).get('SettingsObject', {}).get('@odata.id')
    if set_bios_attr_uri is None:
        return {'ret': False, 'msg': 'Settings resource for BIOS attributes not found.'}
    payload = {'Attributes': attrs_to_patch}
    response = self.patch_request(self.root_uri + set_bios_attr_uri, payload)
    if response['ret'] is False:
        return response
    return {'ret': True, 'changed': True, 'msg': 'Modified BIOS attributes %s' % attrs_to_patch, 'warning': warning}