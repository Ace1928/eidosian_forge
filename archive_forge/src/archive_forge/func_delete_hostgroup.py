from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def delete_hostgroup(module, array):
    changed = True
    try:
        vols = array.list_hgroup_connections(module.params['name'])
    except Exception:
        module.fail_json(msg='Failed to get volume connection for hostgroup {0}'.format(module.params['hostgroup']))
    if not module.check_mode:
        for vol in vols:
            try:
                array.disconnect_hgroup(module.params['name'], vol['vol'])
            except Exception:
                module.fail_json(msg='Failed to disconnect volume {0} from hostgroup {1}'.format(vol['vol'], module.params['name']))
        host = array.get_hgroup(module.params['name'])
        if not module.check_mode:
            try:
                array.set_hgroup(module.params['name'], remhostlist=host['hosts'])
                try:
                    array.delete_hgroup(module.params['name'])
                except Exception:
                    module.fail_json(msg='Failed to delete hostgroup {0}'.format(module.params['name']))
            except Exception:
                module.fail_json(msg='Failed to remove hosts {0} from hostgroup {1}'.format(host['hosts'], module.params['name']))
    module.exit_json(changed=changed)