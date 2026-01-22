from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def set_nioc_enabled(self, state):
    try:
        self.dvs.EnableNetworkResourceManagement(enable=state)
    except vim.fault.DvsFault as dvs_fault:
        self.module.fail_json(msg='DvsFault while setting NIOC enabled=%r: %s' % (state, to_native(dvs_fault.msg)))
    except vim.fault.DvsNotAuthorized as auth_fault:
        self.module.fail_json(msg='Not authorized to set NIOC enabled=%r: %s' % (state, to_native(auth_fault.msg)))
    except vmodl.fault.NotSupported as support_fault:
        self.module.fail_json(msg='NIOC not supported by DVS: %s' % to_native(support_fault.msg))
    except vmodl.RuntimeFault as runtime_fault:
        self.module.fail_json(msg='RuntimeFault while setting NIOC enabled=%r: %s' % (state, to_native(runtime_fault.msg)))