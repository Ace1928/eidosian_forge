from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
def get_tags_for_datastore(self, datastore_mid=None):
    """
        Return list of tag object associated with datastore
        Args:
            datastore_mid: Dynamic object for datacenter

        Returns: List of tag object associated with the given datastore

        """
    dobj = DynamicID(type='Datastore', id=datastore_mid)
    return self.get_tags_for_dynamic_obj(dobj=dobj)