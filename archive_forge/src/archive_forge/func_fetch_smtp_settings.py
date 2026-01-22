from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def fetch_smtp_settings(rest_obj):
    final_resp = rest_obj.invoke_request('GET', SMTP_URL)
    ret_data = final_resp.json_data.get('value')[0]
    ret_data.pop('@odata.type')
    return ret_data