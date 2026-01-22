from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def update_fc_interface(module, array, interface, api_version):
    """Modify FC Interface settings"""
    changed = False
    if FC_ENABLE_API in api_version:
        if not interface.enabled and module.params['state'] == 'present':
            changed = True
            if not module.check_mode:
                network = NetworkInterfacePatch(enabled=True, override_npiv_check=True)
                res = array.patch_network_interfaces(names=[module.params['name']], network=network)
                if res.status_code != 200:
                    module.fail_json(msg='Failed to enable interface {0}.'.format(module.params['name']))
        if interface.enabled and module.params['state'] == 'absent':
            changed = True
            if not module.check_mode:
                network = NetworkInterfacePatch(enabled=False, override_npiv_check=True)
                res = array.patch_network_interfaces(names=[module.params['name']], network=network)
                if res.status_code != 200:
                    module.fail_json(msg='Failed to disable interface {0}.'.format(module.params['name']))
    if module.params['servicelist'] and sorted(module.params['servicelist']) != sorted(interface.services):
        changed = True
        if not module.check_mode:
            network = NetworkInterfacePatch(services=module.params['servicelist'])
            res = array.patch_network_interfaces(names=[module.params['name']], network=network)
            if res.status_code != 200:
                module.fail_json(msg='Failed to update interface service list {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)