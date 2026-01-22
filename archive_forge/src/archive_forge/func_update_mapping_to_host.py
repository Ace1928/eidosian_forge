from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def update_mapping_to_host(module, system):
    """ Update a mapping to a host """
    host = get_host(module, system)
    volume = get_volume(module, system)
    volume_name = module.params['volume']
    host_name = module.params['host']
    desired_lun = module.params['lun']
    if not vol_is_mapped_to_host(volume, host):
        msg = f"Volume '{volume_name}' is not mapped to host '{host_name}'"
        module.fail_json(msg=msg)
    if desired_lun:
        found_lun = find_host_lun(host, volume)
        if found_lun != desired_lun:
            msg = f"Cannot change the lun from '{found_lun}' to '{desired_lun}' for existing mapping of volume '{volume_name}' to host '{host_name}'"
            module.fail_json(msg=msg)
    changed = False
    return changed