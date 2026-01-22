from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def update_vif(module, array, interface, subnet):
    """Modify VLAN Interface settings"""
    changed = False
    vif_info = _get_vif(array, interface, subnet)
    vif_name = vif_info['name']
    if module.params['address']:
        if module.params['address'] != vif_info['address']:
            changed = True
            if not module.check_mode:
                try:
                    array.set_network_interface(vif_name, address=module.params['address'])
                except Exception:
                    module.fail_json(msg='Failed to change IP address for VLAN interface {0}.'.format(subnet))
    if module.params['enabled'] != vif_info['enabled']:
        if module.params['enabled']:
            changed = True
            if not module.check_mode:
                try:
                    array.set_network_interface(vif_name, enabled=True)
                except Exception:
                    module.fail_json(msg='Failed to enable VLAN interface {0}.'.format(vif_name))
        else:
            changed = True
            if not module.check_mode:
                try:
                    array.set_network_interface(vif_name, enabled=False)
                except Exception:
                    module.fail_json(msg='Failed to disable VLAN interface {0}.'.format(vif_name))
    module.exit_json(changed=changed)