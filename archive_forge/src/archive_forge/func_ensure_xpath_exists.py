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
def ensure_xpath_exists(module, tree, xpath, namespaces):
    changed = False
    if not is_node(tree, xpath, namespaces):
        changed = check_or_make_target(module, tree, xpath, namespaces)
    finish(module, tree, xpath, namespaces, changed)