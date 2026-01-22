from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def modify_iface(module, blade):
    """Modify Network Interface IP address"""
    changed = False
    iface = get_iface(module, blade)
    iface_new = []
    iface_new.append(module.params['name'])
    if module.params['address'] != iface.address:
        changed = True
        if not module.check_mode:
            try:
                blade.network_interfaces.update_network_interfaces(names=iface_new, network_interface=NetworkInterface(address=module.params['address']))
                changed = True
            except Exception:
                module.fail_json(msg='Failed to modify Interface {0}'.format(module.params['name']))
    module.exit_json(changed=changed)