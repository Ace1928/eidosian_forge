from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def get_profile_query(rest_obj, query, url_prm):
    prof_list = []
    try:
        if query:
            resp = rest_obj.get_all_items_with_pagination(PROFILE_VIEW, query_param=query)
            prof_list = resp.get('value')
        if url_prm:
            url_resp = rest_obj.invoke_request('GET', '{0}{1}'.format(PROFILE_VIEW, url_prm))
            prof_list = [url_resp.json_data]
    except Exception:
        prof_list = []
    return prof_list