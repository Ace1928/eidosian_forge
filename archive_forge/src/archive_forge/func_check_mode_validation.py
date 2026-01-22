from __future__ import (absolute_import, division, print_function)
import json
import socket
import copy
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def check_mode_validation(module, loc_resp):
    exist_config = {'EnableKvmAccess': loc_resp['EnableKvmAccess'], 'EnableChassisDirect': loc_resp['EnableChassisDirect'], 'EnableChassisPowerButton': loc_resp['EnableChassisPowerButton'], 'EnableLcdOverridePin': loc_resp['EnableLcdOverridePin'], 'LcdAccess': loc_resp['LcdAccess'], 'LcdCustomString': loc_resp['LcdCustomString'], 'LcdLanguage': loc_resp['LcdLanguage']}
    quick_sync = loc_resp['QuickSync']
    exist_quick_config = {'QuickSyncAccess': quick_sync['QuickSyncAccess'], 'TimeoutLimit': quick_sync['TimeoutLimit'], 'EnableInactivityTimeout': quick_sync['EnableInactivityTimeout'], 'TimeoutLimitUnit': quick_sync['TimeoutLimitUnit'], 'EnableReadAuthentication': quick_sync['EnableReadAuthentication'], 'EnableQuickSyncWifi': quick_sync['EnableQuickSyncWifi']}
    req_config, req_quick_config, payload = ({}, {}, {})
    lcd_options, chassis_power = (module.params.get('lcd'), module.params.get('chassis_power_button'))
    if loc_resp['LcdPresence'] == 'Present' and lcd_options is not None:
        req_config['LcdCustomString'] = lcd_options.get('user_defined')
        req_config['LcdAccess'] = lcd_options.get('lcd_access')
        req_config['LcdLanguage'] = lcd_options.get('lcd_language')
    req_config['EnableKvmAccess'] = module.params.get('enable_kvm_access')
    req_config['EnableChassisDirect'] = module.params.get('enable_chassis_direct_access')
    if chassis_power is not None:
        power_button = chassis_power['enable_chassis_power_button']
        if power_button is False:
            chassis_pin = chassis_power.get('enable_lcd_override_pin')
            if chassis_pin is True:
                exist_config['LcdOverridePin'] = loc_resp['LcdOverridePin']
                req_config['LcdOverridePin'] = chassis_power['disabled_button_lcd_override_pin']
            req_config['EnableLcdOverridePin'] = chassis_pin
        req_config['EnableChassisPowerButton'] = power_button
    q_sync = module.params.get('quick_sync')
    if q_sync is not None and loc_resp['QuickSync']['QuickSyncHardware'] == 'Present':
        req_quick_config['QuickSyncAccess'] = q_sync.get('quick_sync_access')
        req_quick_config['EnableReadAuthentication'] = q_sync.get('enable_read_authentication')
        req_quick_config['EnableQuickSyncWifi'] = q_sync.get('enable_quick_sync_wifi')
        if q_sync.get('enable_inactivity_timeout') is True:
            time_limit, time_unit = (q_sync.get('timeout_limit'), q_sync.get('timeout_limit_unit'))
            if q_sync.get('timeout_limit_unit') == 'MINUTES':
                time_limit, time_unit = (time_limit * 60, 'SECONDS')
            req_quick_config['TimeoutLimit'] = time_limit
            req_quick_config['TimeoutLimitUnit'] = time_unit
        req_quick_config['EnableInactivityTimeout'] = q_sync.get('enable_inactivity_timeout')
    req_config = dict([(k, v) for k, v in req_config.items() if v is not None])
    req_quick_config = dict([(k, v) for k, v in req_quick_config.items() if v is not None])
    cloned_req_config = copy.deepcopy(exist_config)
    cloned_req_config.update(req_config)
    cloned_req_quick_config = copy.deepcopy(exist_quick_config)
    cloned_req_quick_config.update(req_quick_config)
    diff_changes = [bool(set(exist_config.items()) ^ set(cloned_req_config.items())) or bool(set(exist_quick_config.items()) ^ set(cloned_req_quick_config.items()))]
    if module.check_mode and any(diff_changes) is True:
        module.exit_json(msg=CHANGES_FOUND, changed=True)
    elif module.check_mode and all(diff_changes) is False or (not module.check_mode and all(diff_changes) is False):
        module.exit_json(msg=NO_CHANGES_FOUND)
    payload.update(cloned_req_config)
    payload['QuickSync'] = cloned_req_quick_config
    payload['QuickSync']['QuickSyncHardware'] = loc_resp['QuickSync']['QuickSyncHardware']
    payload['SettingType'] = 'LocalAccessConfiguration'
    payload['LcdPresence'] = loc_resp['LcdPresence']
    return payload