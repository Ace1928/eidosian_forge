from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def perform_volume_create_modify(module, session_obj):
    """
    perform volume creation and modification for state present
    """
    specified_controller_id = module.params.get('controller_id')
    volume_id = module.params.get('volume_id')
    check_raid_type_supported(module, session_obj)
    action, uri, method = (None, None, None)
    if specified_controller_id is not None:
        check_controller_id_exists(module, session_obj)
        uri = CONTROLLER_VOLUME_URI.format(storage_base_uri=storage_collection_map['storage_base_uri'], controller_id=specified_controller_id)
        method = 'POST'
        action = 'create'
    else:
        resp = check_volume_id_exists(module, session_obj, volume_id)
        if resp.success:
            uri = SETTING_VOLUME_ID_URI.format(storage_base_uri=storage_collection_map['storage_base_uri'], volume_id=volume_id)
            method = 'PATCH'
            action = 'modify'
    payload = volume_payload(module)
    check_mode_validation(module, session_obj, action, uri)
    if not payload:
        module.fail_json(msg='Input options are not provided for the {0} volume task.'.format(action))
    return perform_storage_volume_action(method, uri, session_obj, action, payload)