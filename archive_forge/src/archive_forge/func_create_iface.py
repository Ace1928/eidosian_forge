from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_iface(module, blade):
    """Create Network Interface"""
    changed = True
    if not module.check_mode:
        iface = []
        services = []
        iface.append(module.params['name'])
        services.append(module.params['services'])
        try:
            blade.network_interfaces.create_network_interfaces(names=iface, network_interface=NetworkInterface(address=module.params['address'], services=services, type=module.params['itype']))
        except Exception:
            module.fail_json(msg='Interface creation failed. Check subnet exists for {0}'.format(module.params['address']))
    module.exit_json(changed=changed)