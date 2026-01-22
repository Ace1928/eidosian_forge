from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def get_slot_data(module, rest_obj, ch_slots, chass_id):
    uri = DEVICE_URI + '({0})/DeviceBladeSlots'.format(chass_id)
    chsvc_tag = ch_slots.get('chassis_service_tag')
    resp = rest_obj.invoke_request('GET', uri)
    blade_slots = resp.json_data.get('value')
    if len(blade_slots) < 8:
        resp = get_device_type(rest_obj, 3000)
        storage = resp.get('value')
        for stx in storage:
            if stx.get('ChassisServiceTag') == chsvc_tag:
                blade_slots.append(stx.get('SlotConfiguration'))
    blade_dict = {}
    for slot in blade_slots:
        slot['ChassisId'] = chass_id
        slot['ChassisServiceTag'] = chsvc_tag
        if slot.get('Id'):
            slot['SlotId'] = str(slot.get('Id'))
        blade_dict[slot['SlotNumber']] = slot
        rest_obj.strip_substr_dict(slot)
    inp_slots = ch_slots.get('slots')
    existing_dict = dict([(slot['SlotNumber'], slot['SlotName']) for slot in blade_slots])
    input_dict = dict([(str(slot['slot_number']), slot['slot_name']) for slot in inp_slots])
    invalid_slot_number = set(input_dict.keys()) - set(existing_dict.keys())
    if invalid_slot_number:
        module.fail_json(msg=INVALID_SLOT_NUMBERS.format(';'.join(invalid_slot_number)))
    if len(input_dict) < len(inp_slots):
        module.fail_json(msg=SLOT_NUM_DUP.format(chsvc_tag))
    slot_dict_diff = {}
    slot_diff = recursive_diff(input_dict, existing_dict)
    if slot_diff and slot_diff[0]:
        diff = {}
        for k, v in slot_diff[0].items():
            blade_dict[k]['new_name'] = input_dict.get(k)
            diff['{0}_{1}'.format(chsvc_tag, k)] = blade_dict[k]
        slot_dict_diff.update(diff)
    return slot_dict_diff