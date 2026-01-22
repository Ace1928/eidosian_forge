from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
@classmethod
def valid_state(cls, state):
    """
        A valid state is one of:
            - installed
            - absent
        """
    if state is None:
        return True
    else:
        return isinstance(state, string_types) and state.lower() in ('installed', 'absent')