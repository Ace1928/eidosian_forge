from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
def get_tags_for_hostsystem(self, hostsystem_mid=None):
    """
        Return list of tag object associated with host system
        Args:
            hostsystem_mid: Dynamic object for host system

        Returns: List of tag object associated with the given host system

        """
    dobj = DynamicID(type='HostSystem', id=hostsystem_mid)
    return self.get_tags_for_dynamic_obj(dobj=dobj)