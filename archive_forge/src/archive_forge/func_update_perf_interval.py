from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def update_perf_interval(self, perf_manager, statistic):
    """Update statistics interval"""
    try:
        perf_manager.UpdatePerfInterval(statistic)
    except vmodl.fault.InvalidArgument as invalid_argument:
        self.module.fail_json(msg='The set of arguments passed to the function is not specified correctly or the update does not conform to the rules: %s' % to_native(invalid_argument.msg))