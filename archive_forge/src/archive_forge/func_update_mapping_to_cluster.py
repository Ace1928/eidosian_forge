from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def update_mapping_to_cluster(module, system):
    """ Update a mapping to a cluster """
    cluster = get_cluster(module, system)
    volume = get_volume(module, system)
    desired_lun = module.params['lun']
    volume_name = module.params['volume']
    cluster_name = module.params['cluster']
    if not vol_is_mapped_to_cluster(volume, cluster):
        msg = f'Volume {volume_name} is not mapped to cluster {cluster_name}'
        module.fail_json(msg=msg)
    if desired_lun:
        found_lun = find_cluster_lun(cluster, volume)
        if found_lun != desired_lun:
            msg = f"Cannot change the lun from '{found_lun}' to '{desired_lun}' for existing mapping of volume '{volume_name}' to cluster '{cluster_name}'"
            module.fail_json(msg=msg)
    changed = False
    return changed