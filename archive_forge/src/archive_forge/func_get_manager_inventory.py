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
def get_manager_inventory(self, manager_uri):
    result = {}
    inventory = {}
    properties = ['Id', 'FirmwareVersion', 'ManagerType', 'Manufacturer', 'Model', 'PartNumber', 'PowerState', 'SerialNumber', 'ServiceIdentification', 'Status', 'UUID']
    response = self.get_request(self.root_uri + manager_uri)
    if response['ret'] is False:
        return response
    result['ret'] = True
    data = response['data']
    for property in properties:
        if property in data:
            inventory[property] = data[property]
    result['entries'] = inventory
    return result