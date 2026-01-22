from __future__ import (absolute_import, division, print_function)
import json
import re
import time
from ssl import SSLError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def get_user_account(module, idrac):
    """
    This function gets the slot id and slot uri for create and modify.
    :param module: ansible module arguments
    :param idrac: idrac objects
    :return: user_attr, slot_uri, slot_id, empty_slot, empty_slot_uri
    """
    slot_uri, slot_id, empty_slot, empty_slot_uri = (None, None, None, None)
    if not module.params['user_name']:
        module.fail_json(msg='User name is not valid.')
    response = idrac.export_scp(export_format='JSON', export_use='Default', target='IDRAC', job_wait=True)
    user_attributes = idrac.get_idrac_local_account_attr(response.json_data, fqdd='iDRAC.Embedded.1')
    slot_num = tuple(range(2, 17))
    for num in slot_num:
        user_name = 'Users.{0}#UserName'.format(num)
        if user_attributes.get(user_name) == module.params['user_name']:
            slot_id = num
            slot_uri = ACCOUNT_URI + str(num)
            break
        if not user_attributes.get(user_name) and (empty_slot_uri and empty_slot) is None:
            empty_slot = num
            empty_slot_uri = ACCOUNT_URI + str(num)
    return (user_attributes, slot_uri, slot_id, empty_slot, empty_slot_uri)