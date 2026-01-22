from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from time import sleep
Checking and applying ESXi host configuration one by one,
        from prepared list of hosts in `self.hosts`.
        For every host applied:
        - user input checking done via calling `sanitize_params` method
        - checks hardware compatibility with user input `check_compatibility`
        - conf changes created via `make_diff`
        - changes applied via calling `_update_sriov` method
        - host state before and after via calling `_check_sriov`
        