from __future__ import (absolute_import, division, print_function)
import re
import json
import codecs
import binascii
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def update_fc_settings(payload, settings_params, setting_type, module):
    """payload update for Fibre Channel specific settings
    payload: other setting payload
    settings_params: fc setting parameters
    setting_type: "FcSettings"
    """
    sub_setting_mapper = {}
    starting_address = settings_params.get('starting_address')
    identity_count = settings_params.get('identity_count')
    wwnn_payload = {}
    wwpn_payload = {}
    if starting_address:
        if not mac_validation(starting_address):
            module.fail_json(msg='Please provide the valid starting address format for FC settings.')
        wwnn_prefix, wwpn_prefix = get_wwn_address_prefix(starting_address)
        wwnn_address = mac_to_base64_conversion(wwnn_prefix + starting_address, module)
        wwpn_address = mac_to_base64_conversion(wwpn_prefix + starting_address, module)
        wwnn_payload.update({'StartingAddress': wwnn_address})
        wwpn_payload.update({'StartingAddress': wwpn_address})
    if identity_count is not None:
        wwnn_payload.update({'IdentityCount': identity_count})
        wwpn_payload.update({'IdentityCount': identity_count})
    sub_setting_mapper.update({'Wwnn': wwnn_payload, 'Wwpn': wwpn_payload})
    sub_setting_mapper = dict([(key, val) for key, val in sub_setting_mapper.items() if any(val)])
    if any(sub_setting_mapper):
        payload.update({setting_type: sub_setting_mapper})