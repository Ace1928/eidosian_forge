from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def slot_number_config(module, rest_obj):
    chslots = module.params.get('slot_options')
    resp = get_device_type(rest_obj, 2000)
    chassi_dict = dict([(chx['Identifier'], chx['Id']) for chx in resp.get('value')])
    slot_data = {}
    input_chassi_list = list((chx.get('chassis_service_tag') for chx in chslots))
    duplicate = [x for i, x in enumerate(input_chassi_list) if i != input_chassi_list.index(x)]
    if duplicate:
        module.fail_json(msg=CHASSIS_REPEATED.format(';'.join(set(duplicate))))
    for chx in chslots:
        chsvc_tag = chx.get('chassis_service_tag')
        if chsvc_tag not in chassi_dict.keys():
            module.fail_json(msg=CHASSIS_TAG_INVALID.format(chsvc_tag))
        slot_dict = get_slot_data(module, rest_obj, chx, chassi_dict[chsvc_tag])
        slot_data.update(slot_dict)
    if not slot_data:
        module.exit_json(msg=NO_CHANGES_MSG)
    if module.check_mode:
        module.exit_json(msg=CHANGES_FOUND, changed=True)
    return slot_data