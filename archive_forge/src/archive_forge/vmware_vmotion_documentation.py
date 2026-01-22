from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (

        Find unique virtual machine either by UUID or Name.
        Returns: virtual machine object if found, else None.

        