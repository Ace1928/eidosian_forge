from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def check_physical_disk_exists(module, drives):
    """
    validation to check if physical disks(drives) available for the specified controller
    """
    specified_drives = module.params.get('drives')
    if specified_drives:
        existing_drives = []
        specified_controller_id = module.params.get('controller_id')
        if drives:
            for drive in drives:
                drive_uri = drive['@odata.id']
                drive_id = drive_uri.split('/')[-1]
                existing_drives.append(drive_id)
        else:
            module.fail_json(msg='No Drive(s) are attached to the specified Controller Id: {0}.'.format(specified_controller_id))
        invalid_drives = list(set(specified_drives) - set(existing_drives))
        if invalid_drives:
            invalid_drive_msg = ','.join(invalid_drives)
            module.fail_json(msg='Following Drive(s) {0} are not attached to the specified Controller Id: {1}.'.format(invalid_drive_msg, specified_controller_id))
    return True