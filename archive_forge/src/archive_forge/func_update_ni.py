from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.getters import (
from ansible_collections.purestorage.fusion.plugins.module_utils.networking import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def update_ni(module, fusion, ni):
    """Update Network Interface"""
    ni_api_instance = purefusion.NetworkInterfacesApi(fusion)
    patches = []
    if module.params['display_name'] and module.params['display_name'] != ni.display_name:
        patch = purefusion.NetworkInterfacePatch(display_name=purefusion.NullableString(module.params['display_name']))
        patches.append(patch)
    if module.params['enabled'] is not None and module.params['enabled'] != ni.enabled:
        patch = purefusion.NetworkInterfacePatch(enabled=purefusion.NullableBoolean(module.params['enabled']))
        patches.append(patch)
    if module.params['network_interface_group'] and module.params['network_interface_group'] != ni.network_interface_group:
        if module.params['eth'] and module.params['eth'] != ni.eth:
            patch = purefusion.NetworkInterfacePatch(eth=purefusion.NetworkInterfacePatchEth(purefusion.NullableString(module.params['eth'])), network_interface_group=purefusion.NullableString(module.params['network_interface_group']))
        else:
            patch = purefusion.NetworkInterfacePatch(network_interface_group=purefusion.NullableString(module.params['network_interface_group']))
        patches.append(patch)
    id = None
    if not module.check_mode:
        for patch in patches:
            op = ni_api_instance.update_network_interface(patch, region_name=module.params['region'], availability_zone_name=module.params['availability_zone'], array_name=module.params['array'], net_intf_name=module.params['name'])
            res_op = await_operation(fusion, op)
            id = res_op.result.resource.id
    changed = len(patches) != 0
    module.exit_json(changed=changed, id=id)