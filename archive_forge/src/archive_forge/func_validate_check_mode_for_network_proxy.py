from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def validate_check_mode_for_network_proxy(payload_diff, module):
    """
    check mode support validation
    :param payload_diff: payload difference
    :param module: ansible module object
    :return: None
    """
    if module.check_mode:
        if payload_diff:
            module.exit_json(msg=CHECK_MODE_CHANGE_FOUND_MSG, changed=True)
        else:
            module.exit_json(msg=CHECK_MODE_CHANGE_NOT_FOUND_MSG, changed=False)