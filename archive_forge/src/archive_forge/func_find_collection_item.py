from ansible.module_utils.network.aos.aos import (check_aos_version, get_aos_session, find_collection_item,
from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.network.plugins.module_utils.version import LooseVersion
from ansible.module_utils._text import to_native
def find_collection_item(collection, item_name=False, item_id=False):
    """
    Find collection_item based on name or id from a collection object
    Both Collection_item and Collection Objects are provided by aos-pyez library

    Return
        collection_item: object corresponding to the collection type
    """
    my_dict = None
    if item_name:
        my_dict = collection.find(label=item_name)
    elif item_id:
        my_dict = collection.find(uid=item_id)
    if my_dict is None:
        return collection['']
    else:
        return my_dict