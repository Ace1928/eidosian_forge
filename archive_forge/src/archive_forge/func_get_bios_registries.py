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
def get_bios_registries(self):
    response = self.get_request(self.root_uri + self.systems_uri)
    if not response['ret']:
        return response
    server_details = response['data']
    if 'Bios' not in server_details:
        msg = "Getting BIOS URI failed, Key 'Bios' not found in /redfish/v1/Systems/1/ response: %s"
        return {'ret': False, 'msg': msg % str(server_details)}
    bios_uri = server_details['Bios']['@odata.id']
    bios_resp = self.get_request(self.root_uri + bios_uri)
    if not bios_resp['ret']:
        return bios_resp
    bios_data = bios_resp['data']
    attribute_registry = bios_data['AttributeRegistry']
    reg_uri = self.root_uri + self.service_root + 'Registries/' + attribute_registry
    reg_resp = self.get_request(reg_uri)
    if not reg_resp['ret']:
        return reg_resp
    reg_data = reg_resp['data']
    lst = []
    response = self.check_location_uri(reg_data, reg_uri)
    if not response['ret']:
        return response
    rsp_data, rsp_uri = (response['rsp_data'], response['rsp_uri'])
    if 'RegistryEntries' not in rsp_data:
        return {'msg': "'RegistryEntries' not present in %s response, %s" % (rsp_uri, str(rsp_data)), 'ret': False}
    return {'bios_registry': rsp_data, 'bios_registry_uri': rsp_uri, 'ret': True}