from __future__ import (absolute_import, division, print_function)
import json
import re
import time
from ssl import SSLError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def convert_payload_xml(payload):
    """
    this function converts payload to xml and json data.
    :param payload: user input for payload
    :return: returns xml and json data
    """
    root = '<SystemConfiguration><Component FQDD="iDRAC.Embedded.1">{0}</Component></SystemConfiguration>'
    attr = ''
    json_payload = {}
    for k, v in payload.items():
        key = re.sub('(?<=\\d)\\.', '#', k)
        attr += '<Attribute Name="{0}">{1}</Attribute>'.format(key, v)
        json_payload[key] = v
    root = root.format(attr)
    return (root, json_payload)