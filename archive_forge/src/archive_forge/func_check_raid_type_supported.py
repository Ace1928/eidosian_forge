from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def check_raid_type_supported(module, session_obj):
    volume_type = module.params.get('volume_type')
    if volume_type:
        raid_type = volume_type_map.get(volume_type)
    else:
        raid_type = module.params.get('raid_type')
    if raid_type:
        try:
            specified_controller_id = module.params.get('controller_id')
            uri = CONTROLLER_URI.format(storage_base_uri=storage_collection_map['storage_base_uri'], controller_id=specified_controller_id)
            resp = session_obj.invoke_request('GET', uri)
            supported_raid_types = resp.json_data['StorageControllers'][0]['SupportedRAIDTypes']
            if raid_type not in supported_raid_types:
                module.exit_json(msg=RAID_TYPE_NOT_SUPPORTED_MSG.format(raid_type=raid_type), failed=True)
        except (HTTPError, URLError, SSLValidationError, ConnectionError, TypeError, ValueError) as err:
            raise err