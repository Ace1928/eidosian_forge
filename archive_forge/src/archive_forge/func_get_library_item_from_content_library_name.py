from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
def get_library_item_from_content_library_name(self, name, content_library_name):
    """
        Returns the identifier of the library item with the given name in the specified
        content library.
        Args:
            name (str): The name of item to look for
            content_library_name (str): The name of the content library to search in
        Returns:
            str: The item ID or None if the item is not found
        """
    cl_find_spec = self.api_client.content.Library.FindSpec(name=content_library_name)
    cl_item_ids = self.api_client.content.Library.find(cl_find_spec)
    cl_item_id = cl_item_ids[0] if cl_item_ids else None
    if cl_item_id:
        find_spec = Item.FindSpec(name=name, library_id=cl_item_id)
        item_ids = self.api_client.content.library.Item.find(find_spec)
        item_id = item_ids[0] if item_ids else None
        return item_id
    else:
        return None