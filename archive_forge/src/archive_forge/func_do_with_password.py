from __future__ import absolute_import, division, print_function
import os
import subprocess
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def do_with_password(module, cmd, password):
    env = {}
    if password:
        env = {'PGPASSWORD': password}
    executed_commands.append(cmd)
    rc, stderr, stdout = module.run_command(cmd, use_unsafe_shell=True, environ_update=env)
    return (rc, stderr, stdout, cmd)