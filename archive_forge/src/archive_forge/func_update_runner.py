from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def update_runner(self, runner, arguments):
    changed = False
    for arg_key, arg_value in arguments.items():
        if arguments[arg_key] is not None:
            if isinstance(arguments[arg_key], list):
                list1 = getattr(runner, arg_key)
                list1.sort()
                list2 = arguments[arg_key]
                list2.sort()
                if list1 != list2:
                    setattr(runner, arg_key, arguments[arg_key])
                    changed = True
            elif getattr(runner, arg_key) != arguments[arg_key]:
                setattr(runner, arg_key, arguments[arg_key])
                changed = True
    return (changed, runner)