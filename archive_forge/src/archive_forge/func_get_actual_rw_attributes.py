from __future__ import absolute_import, division, print_function
import json
import re
import sys
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils._text import to_native
def get_actual_rw_attributes(self, filter='name'):
    if self.actual.__class__.count_filtered(self.client, '%s:%s' % (filter, self.attribute_values_dict[filter])) == 0:
        return {}
    server_list = self.actual.__class__.get_filtered(self.client, '%s:%s' % (filter, self.attribute_values_dict[filter]))
    actual_instance = server_list[0]
    ret_val = {}
    for attribute in self.readwrite_attrs:
        if not hasattr(actual_instance, attribute):
            continue
        ret_val[attribute] = getattr(actual_instance, attribute)
    return ret_val