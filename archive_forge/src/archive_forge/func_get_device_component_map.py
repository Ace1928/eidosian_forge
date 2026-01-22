from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_device_component_map(rest_obj, module):
    device_id_tags = _validate_device_attributes(module)
    device_ids, id_tag_map = get_device_ids(rest_obj, module, device_id_tags)
    comps = module.params.get('components')
    dev_comp_map = {}
    if device_ids:
        dev_comp_map = dict([(dev, comps) for dev in device_ids])
    devices = module.params.get('devices')
    if devices:
        for dev in devices:
            if dev.get('id'):
                dev_comp_map[str(dev.get('id'))] = dev.get('components')
            else:
                id = list(id_tag_map.keys())[list(id_tag_map.values()).index(dev.get('service_tag'))]
                dev_comp_map[str(id)] = dev.get('components')
    return dev_comp_map