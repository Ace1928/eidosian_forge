from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from urllib.parse import quote
import copy
import traceback
def validate_mkey(params):
    selector = params['selector']
    selector_params = params.get('params', {})
    if selector not in MODULE_MKEY_DEFINITONS:
        return (False, {'message': 'unknown selector: ' + selector})
    definition = MODULE_MKEY_DEFINITONS.get(selector, {})
    if not selector_params or len(selector_params) == 0 or len(definition) == 0:
        return (True, {})
    mkey = definition['mkey']
    mkey_type = definition['mkey_type']
    if mkey_type is None:
        return (False, {'message': 'params are not allowed for ' + selector})
    mkey_value = selector_params.get(mkey)
    if not mkey_value:
        return (False, {'message': "param '" + mkey + "' is required"})
    if not isinstance(mkey_value, mkey_type):
        return (False, {'message': "param '" + mkey + "' does not match, " + str(mkey_type) + ' required'})
    return (True, {})