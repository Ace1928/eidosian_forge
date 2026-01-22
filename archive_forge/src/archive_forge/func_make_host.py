from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def make_host(module, array):
    """Create a new host"""
    changed = True
    if not module.check_mode:
        try:
            array.create_host(module.params['name'])
        except Exception:
            module.fail_json(msg='Host {0} creation failed.'.format(module.params['name']))
        try:
            if module.params['vlan']:
                _set_vlan(module)
            _set_host_initiators(module, array)
            api_version = array._list_available_rest_versions()
            if AC_REQUIRED_API_VERSION in api_version and module.params['personality']:
                _set_host_personality(module, array)
            if PREFERRED_ARRAY_API_VERSION in api_version and module.params['preferred_array']:
                _set_preferred_array(module, array)
            if module.params['host_user'] or module.params['target_user']:
                _set_chap_security(module, array)
            if module.params['volume']:
                if module.params['lun']:
                    array.connect_host(module.params['name'], module.params['volume'], lun=module.params['lun'])
                else:
                    array.connect_host(module.params['name'], module.params['volume'])
        except Exception:
            module.fail_json(msg='Host {0} configuration failed.'.format(module.params['name']))
    module.exit_json(changed=changed)