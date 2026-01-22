from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
import json
def validator_is_different(client, db, collection, required, properties, action, level):
    is_different = False
    validator = get_validator(client, db, collection)
    if validator is not None:
        if sorted(required) != sorted(validator.get('required', [])):
            is_different = True
        if action != validator.get('validationAction', 'error'):
            is_different = True
        if level != validator.get('validationLevel', 'strict'):
            is_different = True
        dict1 = json.dumps(properties, sort_keys=True)
        dict2 = json.dumps(validator.get('properties', {}), sort_keys=True)
        if dict1 != dict2:
            is_different = True
    else:
        is_different = True
    return is_different