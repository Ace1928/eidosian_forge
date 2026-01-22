from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_manager(module, blade):
    """Create SNMP Manager"""
    changed = True
    if not module.check_mode:
        if not module.params['version']:
            module.fail_json(msg='SNMP version required to create a new manager')
        if module.params['version'] == 'v2c':
            v2_attrs = SnmpV2c(community=module.params['community'])
            new_v2_manager = SnmpManager(host=module.params['host'], notification=module.params['notification'], version='v2c', v2c=v2_attrs)
            try:
                blade.snmp_managers.create_snmp_managers(names=[module.params['name']], snmp_manager=new_v2_manager)
            except Exception:
                module.fail_json(msg='Failed to create v2c SNMP manager {0}.'.format(module.params['name']))
        else:
            v3_attrs = SnmpV3(auth_protocol=module.params['auth_protocol'], auth_passphrase=module.params['auth_passphrase'], privacy_protocol=module.params['privacy_protocol'], privacy_passphrase=module.params['privacy_passphrase'], user=module.params['user'])
            new_v3_manager = SnmpManager(host=module.params['host'], notification=module.params['notification'], version='v3', v3=v3_attrs)
            try:
                blade.snmp_managers.create_snmp_managers(names=[module.params['name']], snmp_manager=new_v3_manager)
            except Exception:
                module.fail_json(msg='Failed to create v3 SNMP manager {0}.'.format(module.params['name']))
    module.exit_json(changed=changed)