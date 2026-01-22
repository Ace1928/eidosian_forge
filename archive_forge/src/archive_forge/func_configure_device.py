from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
def configure_device(module, warnings, candidate):
    kwargs = {}
    config_format = None
    if module.params['src']:
        config_format = module.params['src_format'] or guess_format(str(candidate))
        if config_format == 'set':
            kwargs.update({'format': 'text', 'action': 'set'})
        else:
            kwargs.update({'format': config_format, 'action': module.params['update']})
    if isinstance(candidate, string_types):
        candidate = candidate.split('\n')
    if any((module.params['lines'], config_format == 'set')):
        candidate = filter_delete_statements(module, candidate)
        kwargs['format'] = 'text'
        kwargs['action'] = 'set'
    return load_config(module, candidate, warnings, **kwargs)