from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def get_ome_group_by_id(rest_obj, id):
    grp = {}
    try:
        resp = rest_obj.invoke_request('GET', GROUP_URI + '({0})'.format(id))
        grp = resp.json_data
    except Exception:
        grp = {}
    return grp