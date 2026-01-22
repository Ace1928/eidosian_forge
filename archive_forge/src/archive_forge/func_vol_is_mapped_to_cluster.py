from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def vol_is_mapped_to_cluster(volume, cluster):
    """ Return a bool showing if a vol is mapped to a cluster """
    cluster_luns = cluster.get_luns()
    for lun in cluster_luns:
        if lun.volume == volume:
            return True
    return False