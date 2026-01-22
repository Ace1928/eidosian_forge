from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_kmip_dict(array):
    kmip_info = {}
    api_version = array._list_available_rest_versions()
    if P53_API_VERSION in api_version:
        kmips = array.list_kmip()
        for kmip in range(0, len(kmips)):
            key = kmips[kmip]['name']
            kmip_info[key] = {'certificate': kmips[kmip]['certificate'], 'ca_cert_configured': kmips[kmip]['ca_certificate_configured'], 'uri': kmips[kmip]['uri']}
    return kmip_info