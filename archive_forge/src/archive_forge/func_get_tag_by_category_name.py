from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
def get_tag_by_category_name(self, tag_name=None, category_name=None):
    """
        Return tag object by category name
        Args:
            tag_name: Name of tag
            category_id: Id of category
        Returns: Tag object if found else None
        """
    category_id = None
    if category_name is not None:
        category_obj = self.get_category_by_name(category_name=category_name)
        if category_obj is not None:
            category_id = category_obj.id
    return self.get_tag_by_category_id(tag_name=tag_name, category_id=category_id)