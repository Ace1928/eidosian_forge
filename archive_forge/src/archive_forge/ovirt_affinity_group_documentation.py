from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (

        This method iterate via the affinity VM assignments and datech the VMs
        which should not be attached to affinity and attach VMs which should be
        attached to affinity.
        