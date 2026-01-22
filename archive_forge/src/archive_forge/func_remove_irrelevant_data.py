from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.api import (
from ansible_collections.community.routeros.plugins.module_utils._api_data import (
def remove_irrelevant_data(entry, path_info):
    for k, v in list(entry.items()):
        if k == '.id':
            continue
        if k not in path_info.fields or v is None:
            del entry[k]