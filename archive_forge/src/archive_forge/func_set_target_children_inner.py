from __future__ import (absolute_import, division, print_function)
import copy
import json
import os
import re
import traceback
from io import BytesIO
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, json_dict_bytes_to_unicode, missing_required_lib
from ansible.module_utils.six import iteritems, string_types
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.common._collections_compat import MutableMapping
def set_target_children_inner(module, tree, xpath, namespaces, children, in_type):
    matches = tree.xpath(xpath, namespaces=namespaces)
    children = children_to_nodes(module, children, in_type)
    children_as_string = [etree.tostring(c) for c in children]
    changed = False
    for match in matches:
        if len(list(match)) == len(children):
            for idx, element in enumerate(list(match)):
                if etree.tostring(element) != children_as_string[idx]:
                    replace_children_of(children, match)
                    changed = True
                    break
        else:
            replace_children_of(children, match)
            changed = True
    return changed