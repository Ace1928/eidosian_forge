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
def update_iscsi_specific_settings(payload, settings_params, setting_type):
    """payload update for Iscsi specific settings"""
    sub_setting_mapper = {}
    initiator_config = settings_params.get('initiator_config')
    if initiator_config and initiator_config.get('iqn_prefix'):
        sub_setting_mapper.update({'InitiatorConfig': {'IqnPrefix': initiator_config.get('iqn_prefix')}})
    if settings_params.get('initiator_ip_pool_settings'):
        initiator_ip_pool_settings = settings_params['initiator_ip_pool_settings']
        initiator_ip_pool_settings = {'IpRange': initiator_ip_pool_settings.get('ip_range'), 'SubnetMask': initiator_ip_pool_settings.get('subnet_mask'), 'Gateway': initiator_ip_pool_settings.get('gateway'), 'PrimaryDnsServer': initiator_ip_pool_settings.get('primary_dns_server'), 'SecondaryDnsServer': initiator_ip_pool_settings.get('secondary_dns_server')}
        initiator_ip_pool_settings = dict([(k, v) for k, v in initiator_ip_pool_settings.items() if v is not None])
        sub_setting_mapper.update({'InitiatorIpPoolSettings': initiator_ip_pool_settings})
    if any(sub_setting_mapper):
        if 'IscsiSettings' in payload:
            'update MAC address setting'
            sub_setting_mapper.update(payload[setting_type])
        sub_setting_mapper = dict([(key, val) for key, val in sub_setting_mapper.items() if any(val)])
        payload.update({setting_type: sub_setting_mapper})