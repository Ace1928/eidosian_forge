from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_item_and_list
def validate_ioms(module, rest_obj, uplinks):
    uplinkports = get_all_uplink_ports(uplinks)
    payload_ports = []
    occupied_ports = []
    used_ports = []
    for idx in uplinkports:
        used_ports.append(idx['Id'])
    iomsts = ('primary', 'secondary')
    for iom in iomsts:
        prim_st = module.params.get(iom + '_switch_service_tag')
        if prim_st:
            prim_ports = list((str(port).strip() for port in module.params.get(iom + '_switch_ports')))
            id, ioms = get_item_id(rest_obj, prim_st, IOM_DEVICES, key='DeviceServiceTag')
            if not id:
                module.fail_json(msg='Device with service tag {0} does not exist.'.format(prim_st))
            resp = rest_obj.invoke_request('GET', PORT_INFO.format(device_id=id))
            port_info_data = resp.json_data.get('InventoryInfo', [])
            port_info_list = []
            for port in port_info_data:
                if port.get('SubPorts'):
                    for subport in port.get('SubPorts'):
                        port_info_list.append(subport['PortNumber'])
                else:
                    port_info_list.append(port['PortNumber'])
            non_exist_ports = []
            for port in prim_ports:
                if port not in port_info_list:
                    non_exist_ports.append(port)
                st_port = prim_st + ':' + port
                payload_ports.append(st_port)
                if st_port in used_ports:
                    occupied_ports.append(st_port)
            if non_exist_ports:
                module.fail_json(msg='{0} Port Numbers {1} does not exist for IOM {2}.'.format(iom, ','.join(set(non_exist_ports)), prim_st))
    if occupied_ports:
        module.fail_json(msg='Ports {0} are already occupied.'.format(','.join(set(occupied_ports))))
    return payload_ports