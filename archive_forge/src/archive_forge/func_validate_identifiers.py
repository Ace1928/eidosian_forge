from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
def validate_identifiers(available_values, requested_values, identifier_types, module):
    """
    Validate if requested group/device ids are valid
    """
    val = set(requested_values) - set(available_values)
    if val:
        module.fail_json(msg=INVALID_IDENTIFIER.format(identifier=identifier_types, invalid_val=','.join(map(str, val))))