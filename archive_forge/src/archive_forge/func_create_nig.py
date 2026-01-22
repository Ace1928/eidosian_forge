from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.networking import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def create_nig(module, fusion):
    """Create Network Interface Group"""
    nig_api_instance = purefusion.NetworkInterfaceGroupsApi(fusion)
    changed = False
    if module.params['gateway'] and (not is_address_in_network(module.params['gateway'], module.params['prefix'])):
        module.fail_json(msg='`gateway` must be an address in subnet `prefix`')
    id = None
    if not module.check_mode:
        display_name = module.params['display_name'] or module.params['name']
        if module.params['group_type'] == 'eth':
            if module.params['gateway']:
                eth = purefusion.NetworkInterfaceGroupEthPost(prefix=module.params['prefix'], gateway=module.params['gateway'], mtu=module.params['mtu'])
            else:
                eth = purefusion.NetworkInterfaceGroupEthPost(prefix=module.params['prefix'], mtu=module.params['mtu'])
            nig = purefusion.NetworkInterfaceGroupPost(group_type='eth', eth=eth, name=module.params['name'], display_name=display_name)
            op = nig_api_instance.create_network_interface_group(nig, availability_zone_name=module.params['availability_zone'], region_name=module.params['region'])
            res_op = await_operation(fusion, op)
            id = res_op.result.resource.id
            changed = True
        else:
            module.warn(f'group_type={module.params['group_type']} is not implemented')
    module.exit_json(changed=changed, id=id)