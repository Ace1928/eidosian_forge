from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def create_vif(module, array, interface, subnet):
    """Create VLAN Interface"""
    changed = True
    if not module.check_mode:
        vif_name = interface['name'] + '.' + str(subnet['vlan'])
        if module.params['address']:
            try:
                array.create_vlan_interface(vif_name, module.params['subnet'], address=module.params['address'])
            except Exception:
                module.fail_json(msg='Failed to create VLAN interface {0}.'.format(vif_name))
        else:
            try:
                array.create_vlan_interface(vif_name, module.params['subnet'])
            except Exception:
                module.fail_json(msg='Failed to create VLAN interface {0}.'.format(vif_name))
        if not module.params['enabled']:
            try:
                array.set_network_interface(vif_name, enabled=False)
            except Exception:
                module.fail_json(msg='Failed to disable VLAN interface {0} on creation.'.format(vif_name))
    module.exit_json(changed=changed)