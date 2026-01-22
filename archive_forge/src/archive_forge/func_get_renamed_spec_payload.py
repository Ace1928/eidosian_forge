from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import env_fallback
from . import client
def get_renamed_spec_payload(source, param_mapping):
    return dict(((n, source[k]) for k, n in param_mapping.items() if source.get(k) is not None))