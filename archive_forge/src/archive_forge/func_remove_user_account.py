from __future__ import (absolute_import, division, print_function)
import json
import re
import time
from ssl import SSLError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def remove_user_account(module, idrac, slot_uri, slot_id):
    """
    remove user user account by passing empty payload details.
    :param module: user account module arguments.
    :param idrac: idrac object.
    :param slot_uri: user slot uri.
    :param slot_id: user slot id.
    :return: json.
    """
    response, msg = ({}, 'Successfully deleted user account.')
    payload = get_payload(module, slot_id, action='delete')
    xml_payload, json_payload = convert_payload_xml(payload)
    if module.check_mode and (slot_id and slot_uri) is not None:
        module.exit_json(msg='Changes found to commit!', changed=True)
    elif module.check_mode and (slot_uri and slot_id) is None:
        module.exit_json(msg='No changes found to commit!')
    elif not module.check_mode and (slot_uri and slot_id) is not None:
        time.sleep(10)
        response = idrac.import_scp(import_buffer=xml_payload, target='ALL', job_wait=True)
    else:
        module.exit_json(msg='The user account is absent.')
    return (response, msg)