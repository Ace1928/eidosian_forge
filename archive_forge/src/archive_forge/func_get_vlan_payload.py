from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_vlan_payload(module, rest_obj, untag_dict, tagged_dict):
    payload = {}
    template = get_template_details(module, rest_obj)
    payload['TemplateId'] = template['Id']
    payload['IdentityPoolId'] = template['IdentityPoolId']
    port_id_map, port_untagged_map, port_tagged_map, port_nic_bond_map, nic_bonding_tech = get_template_vlan_info(module, rest_obj, template['Id'])
    payload['BondingTechnology'] = nic_bonding_tech
    payload['PropagateVlan'] = module.params.get('propagate_vlan')
    untag_equal_dict = compare_nested_dict(untag_dict, port_untagged_map)
    tag_equal_dict = compare_nested_dict(tagged_dict, port_tagged_map)
    if untag_equal_dict and tag_equal_dict:
        module.exit_json(msg=NO_CHANGES_MSG)
    vlan_attributes = []
    for pk, pv in port_id_map.items():
        mdict = {}
        if pk in untag_dict or pk in tagged_dict:
            mdict['Untagged'] = untag_dict.pop(pk, port_untagged_map.get(pk))
            mdict['Tagged'] = tagged_dict.pop(pk, port_tagged_map.get(pk))
            mdict['ComponentId'] = port_id_map.get(pk)
            mdict['IsNicBonded'] = port_nic_bond_map.get(pk)
        if mdict:
            vlan_attributes.append(mdict)
    if untag_dict:
        module.fail_json(msg='Invalid port(s) {0} found for untagged VLAN'.format(untag_dict.keys()))
    if tagged_dict:
        module.fail_json(msg='Invalid port(s) {0} found for tagged VLAN'.format(tagged_dict.keys()))
    if module.check_mode:
        module.exit_json(changed=True, msg=CHANGES_FOUND)
    payload['VlanAttributes'] = vlan_attributes
    return payload