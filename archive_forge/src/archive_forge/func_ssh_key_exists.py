from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def ssh_key_exists(self, user, sshkey_name):
    return any((k.title == sshkey_name for k in user.keys.list(**list_all_kwargs)))