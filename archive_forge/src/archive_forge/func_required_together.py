from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import env_fallback
def required_together():
    """Return the default list used for the required_together argument to AnsibleModule"""
    return [['user', 'password']]