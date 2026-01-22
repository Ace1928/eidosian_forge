from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def match_resource_limits(module, current, desired):
    """Check and match limits.

    Args:
        module (AnsibleModule): Ansible module object.
        current (dict): Dictionary with current limits.
        desired (dict): Dictionary with desired limits.

    Returns: Dictionary containing parameters that need to change.
    """
    if not current:
        return desired
    needs_to_change = {}
    for key, val in iteritems(desired):
        if key not in current:
            module.fail_json(msg="resource_limits: key '%s' is unsupported." % key)
        try:
            val = int(val)
        except Exception:
            module.fail_json(msg="Can't convert value '%s' to integer." % val)
        if val != current.get(key):
            needs_to_change[key] = val
    return needs_to_change