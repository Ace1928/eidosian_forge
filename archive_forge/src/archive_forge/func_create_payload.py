from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def create_payload(module, curr_payload):
    console_setting_list = []
    updated_payload = {'ConsoleSetting': []}
    payload_dict = create_payload_dict(curr_payload)
    get_sid = module.params.get('server_initiated_discovery')
    get_ds = module.params.get('discovery_settings')
    get_mcs = module.params.get('metrics_collection_settings')
    get_email = module.params.get('email_sender_settings')
    get_tff = module.params.get('trap_forwarding_format')
    get_mx = module.params.get('mx7000_onboarding_preferences')
    get_rrl = module.params.get('report_row_limit')
    get_dh = module.params.get('device_health')
    get_bas = module.params.get('builtin_appliance_share')
    if get_mcs:
        payload1 = payload_dict['DATA_PURGE_INTERVAL'].copy()
        payload1['Value'] = get_mcs
        console_setting_list.append(payload1)
    if get_email:
        payload2 = payload_dict['EMAIL_SENDER'].copy()
        payload2['Value'] = get_email
        console_setting_list.append(payload2)
    if get_tff:
        dict1 = {'Original': 'AsIs', 'Normalized': 'Normalized'}
        payload3 = payload_dict['TRAP_FORWARDING_SETTING'].copy()
        payload3['Value'] = dict1.get(get_tff)
        console_setting_list.append(payload3)
    if get_mx:
        payload4 = payload_dict['MX7000_ONBOARDING_PREF'].copy()
        payload4['Value'] = get_mx
        console_setting_list.append(payload4)
    if get_rrl:
        payload5 = payload_dict['REPORTS_MAX_RESULTS_LIMIT'].copy()
        payload5['Value'] = get_rrl
        console_setting_list.append(payload5)
    if get_sid:
        if get_sid.get('device_discovery_approval_policy'):
            payload6 = payload_dict['DISCOVERY_APPROVAL_POLICY'].copy()
            payload6['Value'] = get_sid.get('device_discovery_approval_policy')
            console_setting_list.append(payload6)
        if get_sid.get('set_trap_destination') is not None:
            payload7 = payload_dict['NODE_INITIATED_DISCOVERY_SET_TRAP_DESTINATION'].copy()
            payload7['Value'] = get_sid.get('set_trap_destination')
            console_setting_list.append(payload7)
    if get_ds:
        if get_ds.get('general_device_naming') and get_ds.get('server_device_naming'):
            value = 'PREFER_' + module.params['discovery_settings']['general_device_naming'] + ',' + 'PREFER_' + get_ds['server_device_naming']
            payload8 = payload_dict['DEVICE_PREFERRED_NAME'].copy()
            payload8['Value'] = value
            console_setting_list.append(payload8)
        elif get_ds.get('general_device_naming'):
            payload9 = payload_dict['DEVICE_PREFERRED_NAME'].copy()
            payload9['Value'] = 'PREFER_' + get_ds['general_device_naming']
            console_setting_list.append(payload9)
        elif get_ds.get('server_device_naming'):
            payload10 = payload_dict['DEVICE_PREFERRED_NAME'].copy()
            payload10['Value'] = 'PREFER_' + get_ds['server_device_naming']
            console_setting_list.append(payload10)
        if get_ds.get('invalid_device_hostname'):
            payload11 = payload_dict['INVALID_DEVICE_HOSTNAME'].copy()
            payload11['Value'] = get_ds.get('invalid_device_hostname')
            console_setting_list.append(payload11)
        if get_ds.get('common_mac_addresses'):
            payload12 = payload_dict['COMMON_MAC_ADDRESSES'].copy()
            payload12['Value'] = get_ds.get('common_mac_addresses')
            console_setting_list.append(payload12)
    if get_dh and get_dh.get('health_and_power_state_on_connection_lost'):
        payload13 = payload_dict['CONSOLE_CONNECTION_SETTING'].copy()
        payload13['Value'] = get_dh.get('health_and_power_state_on_connection_lost')
        console_setting_list.append(payload13)
    if get_bas and get_bas.get('share_options') == 'CIFS':
        payload14 = payload_dict['MIN_PROTOCOL_VERSION'].copy()
        payload14['Value'] = get_bas.get('cifs_options')
        console_setting_list.append(payload14)
    updated_payload['ConsoleSetting'] = console_setting_list
    return (updated_payload, payload_dict)