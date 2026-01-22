from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
def get_tags_for_vm(self, vm_mid=None):
    """
        Return list of tag object associated with virtual machine
        Args:
            vm_mid: Dynamic object for virtual machine

        Returns: List of tag object associated with the given virtual machine

        """
    dobj = DynamicID(type='VirtualMachine', id=vm_mid)
    return self.get_tags_for_dynamic_obj(dobj=dobj)