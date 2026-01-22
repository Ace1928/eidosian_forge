from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def convert_limits_in_kbps(self, network_limits):
    """
        Convert the limits into KBps

        :param network_limits: dict containing all Network bandwidth limits
        :rtype: converted network limits
        """
    limit_params = ['rebuild_limit', 'rebalance_limit', 'vtree_migration_limit', 'overall_limit']
    modified_limits = dict()
    modified_limits['rebuild_limit'] = None
    modified_limits['rebalance_limit'] = None
    modified_limits['vtree_migration_limit'] = None
    modified_limits['overall_limit'] = None
    if network_limits is None:
        return None
    for limits in network_limits:
        if network_limits[limits] is not None and limits in limit_params:
            if network_limits['bandwidth_unit'] == 'GBps':
                modified_limits[limits] = network_limits[limits] * 1024 * 1024
            elif network_limits['bandwidth_unit'] == 'MBps':
                modified_limits[limits] = network_limits[limits] * 1024
            else:
                modified_limits[limits] = network_limits[limits]
    return modified_limits