from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_obj
from ansible.module_utils._text import to_native
@staticmethod
def get_previous_targets(trap_targets):
    """Get target entries from trap targets object"""
    previous_targets = []
    for target in trap_targets:
        temp = dict()
        temp['hostname'] = target.hostName
        temp['port'] = target.port
        temp['community'] = target.community
        previous_targets.append(temp)
    return previous_targets