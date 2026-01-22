from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def perform_volume_deletion(module, session_obj):
    """
    perform volume deletion for state absent
    """
    volume_id = module.params.get('volume_id')
    if volume_id:
        resp = check_volume_id_exists(module, session_obj, volume_id)
        if hasattr(resp, 'success') and resp.success and (not module.check_mode):
            uri = VOLUME_ID_URI.format(storage_base_uri=storage_collection_map['storage_base_uri'], volume_id=volume_id)
            method = 'DELETE'
            return perform_storage_volume_action(method, uri, session_obj, 'delete')
        elif hasattr(resp, 'success') and resp.success and module.check_mode:
            module.exit_json(msg=CHANGES_FOUND, changed=True)
        elif hasattr(resp, 'code') and resp.code == 404 and module.check_mode:
            module.exit_json(msg=NO_CHANGES_FOUND)
    else:
        module.fail_json(msg="'volume_id' option is a required property for deleting a volume.")