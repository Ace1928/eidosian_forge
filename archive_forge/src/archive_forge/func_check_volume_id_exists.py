from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def check_volume_id_exists(module, session_obj, volume_id):
    """
    validation to check if volume id is valid in case of modify, delete, initialize operation
    """
    uri = VOLUME_ID_URI.format(storage_base_uri=storage_collection_map['storage_base_uri'], volume_id=volume_id)
    err_message = 'Specified Volume Id {0} does not exist in the System.'.format(volume_id)
    resp = check_specified_identifier_exists_in_the_system(module, session_obj, uri, err_message)
    return resp