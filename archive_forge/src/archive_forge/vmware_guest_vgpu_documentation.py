from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (

        Get matched vGPU profile object of ESXi host
        Args:
            vm_obj: Managed object of virtual machine
            vgpu_prfl: vGPU profile name
        Returns: vGPU profile object
        