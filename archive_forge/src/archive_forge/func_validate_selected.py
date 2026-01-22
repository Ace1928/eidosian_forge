from __future__ import (absolute_import, division, print_function)
import uuid
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.common.text.converters import to_native
def validate_selected(l, resource_type, spec):
    if len(l) > 1:
        _msg = 'more than one {0} matches specification {1}: {2}'.format(resource_type, spec, l)
        raise Exception(_msg)
    if len(l) == 0:
        _msg = 'no {0} matches specification: {1}'.format(resource_type, spec)
        raise Exception(_msg)