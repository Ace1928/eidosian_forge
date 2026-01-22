from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.collections import is_string
from ansible.module_utils.six import iteritems
def copy_identity_properties(source_obj, dest_obj):
    for property_name in IDENTITY_PROPERTIES:
        if property_name in source_obj:
            dest_obj[property_name] = source_obj[property_name]
    return dest_obj