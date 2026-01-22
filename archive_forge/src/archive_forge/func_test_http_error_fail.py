from __future__ import (absolute_import, division, print_function)
import json
import os
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def test_http_error_fail(module, err):
    try:
        error_info = json.load(err)
        err_list = error_info.get('error', {}).get('@Message.ExtendedInfo', [ERR_READ_FAIL])
        if err_list:
            err_rsn = err_list[0].get('Message')
    except Exception:
        err_rsn = ERR_READ_FAIL
    module.fail_json(msg='{0}{1}'.format(TEST_CONNECTION_FAIL, err_rsn), error_info=error_info)