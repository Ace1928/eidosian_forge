from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def construct_tree_str(nprfx, attr_detailed):
    str_lst = nprfx.split(SEPRTR)
    br = attr_detailed
    for xs in str_lst:
        if xs not in br:
            br[xs] = {}
        br = br.get(xs)
    return br