from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def check_domain_service(module, rest_obj):
    try:
        rest_obj.invoke_request('GET', DOMAIN_URI, api_timeout=5)
    except HTTPError as err:
        err_message = json.load(err)
        if err_message['error']['@Message.ExtendedInfo'][0]['MessageId'] == 'CGEN1006':
            module.fail_json(msg=DOMAIN_FAIL_MSG)
    return