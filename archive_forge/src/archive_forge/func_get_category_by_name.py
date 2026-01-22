from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
def get_category_by_name(self, category_name=None):
    """
        Return category object by name
        Args:
            category_name: Name of category

        Returns: Category object if found else None
        """
    if not category_name:
        return None
    return self.search_svc_object_by_name(service=self.api_client.tagging.Category, svc_obj_name=category_name)