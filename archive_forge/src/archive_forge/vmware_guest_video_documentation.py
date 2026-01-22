from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task

        Reconfigure video card settings of virtual machine
        Args:
            vm_obj: Managed object of virtual machine
        Returns: Reconfigure results
        