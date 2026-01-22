from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def get_powerflex_protection_domain_parameters():
    """This method provides parameters required for the protection domain
    module on PowerFlex"""
    return dict(protection_domain_name=dict(), protection_domain_new_name=dict(), protection_domain_id=dict(), is_active=dict(type='bool'), network_limits=dict(type='dict', options=dict(rebuild_limit=dict(type='int'), rebalance_limit=dict(type='int'), vtree_migration_limit=dict(type='int'), overall_limit=dict(type='int'), bandwidth_unit=dict(choices=['KBps', 'MBps', 'GBps'], default='KBps'))), rf_cache_limits=dict(type='dict', options=dict(is_enabled=dict(type='bool'), page_size=dict(type='int'), max_io_limit=dict(type='int'), pass_through_mode=dict(choices=['None', 'Read', 'Write', 'ReadAndWrite', 'WriteMiss']))), state=dict(required=True, type='str', choices=['present', 'absent']))