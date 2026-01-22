from __future__ import absolute_import, division, print_function
import os
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils._stormssh import ConfigParser, HAS_PARAMIKO, PARAMIKO_IMPORT_ERROR
from ansible_collections.community.general.plugins.module_utils.ssh import determine_config_file
@staticmethod
def change_host(options, **kwargs):
    options = deepcopy(options)
    changed = False
    for k, v in kwargs.items():
        if '_' in k:
            k = k.replace('_', '')
        if not v:
            if options.get(k):
                del options[k]
                changed = True
        elif options.get(k) != v and (not (isinstance(options.get(k), list) and v in options.get(k))):
            options[k] = v
            changed = True
    return (changed, options)