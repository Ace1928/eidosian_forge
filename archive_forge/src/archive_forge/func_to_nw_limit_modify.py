from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def to_nw_limit_modify(pd_details, network_limits):
    """
    Check if modification required network bandwidth limit for protection
    domain
    :param pd_details: Details of the protection domain
    :type pd_details: dict
    :param network_limits: dict of Network bandwidth limit
    :type network_limits: dict
    :return: Dictionary containing the attributes of protection domain
             which are to be updated
    :rtype: dict
    """
    modify_dict = {}
    if network_limits is not None:
        modify_dict['rebuild_limit'] = None
        modify_dict['rebalance_limit'] = None
        modify_dict['vtree_migration_limit'] = None
        modify_dict['overall_limit'] = None
        if network_limits['rebuild_limit'] is not None and pd_details['rebuildNetworkThrottlingInKbps'] != network_limits['rebuild_limit']:
            modify_dict['rebuild_limit'] = network_limits['rebuild_limit']
        if network_limits['rebalance_limit'] is not None and pd_details['rebalanceNetworkThrottlingInKbps'] != network_limits['rebalance_limit']:
            modify_dict['rebalance_limit'] = network_limits['rebalance_limit']
        if network_limits['vtree_migration_limit'] is not None and pd_details['vtreeMigrationNetworkThrottlingInKbps'] != network_limits['vtree_migration_limit']:
            modify_dict['vtree_migration_limit'] = network_limits['vtree_migration_limit']
        if network_limits['overall_limit'] is not None and pd_details['overallIoNetworkThrottlingInKbps'] != network_limits['overall_limit']:
            modify_dict['overall_limit'] = network_limits['overall_limit']
    return modify_dict