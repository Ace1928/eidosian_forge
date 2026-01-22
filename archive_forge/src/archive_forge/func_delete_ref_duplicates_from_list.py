from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.collections import is_string
from ansible.module_utils.six import iteritems
def delete_ref_duplicates_from_list(refs):
    if all((type(i) is dict and is_object_ref(i) for i in refs)):
        unique_refs = set()
        unique_list = list()
        for i in refs:
            key = (i['id'], i['type'])
            if key not in unique_refs:
                unique_refs.add(key)
                unique_list.append(i)
        return list(unique_list)
    else:
        return refs