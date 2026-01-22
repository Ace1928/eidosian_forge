from __future__ import absolute_import, division, print_function
import filecmp
import os
import re
import shlex
import stat
import sys
import shutil
import tempfile
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.six import b, string_types
def set_git_ssh_env(key_file, ssh_opts, git_version, module):
    """
        use environment variables to configure git's ssh execution,
        which varies by version but this functino should handle all.
    """
    if ssh_opts is None:
        ssh_opts = os.environ.get('GIT_SSH_OPTS', '')
    else:
        ssh_opts = os.environ.get('GIT_SSH_OPTS', '') + ' ' + ssh_opts
    accept_key = 'StrictHostKeyChecking=no'
    if module.params['accept_hostkey'] and accept_key not in ssh_opts:
        ssh_opts += ' -o %s' % accept_key
    force_batch = 'BatchMode=yes'
    if force_batch not in ssh_opts:
        ssh_opts += ' -o %s' % force_batch
    if key_file:
        key_opt = '-i %s' % key_file
        if key_opt not in ssh_opts:
            ssh_opts += '  %s' % key_opt
        ikey = 'IdentitiesOnly=yes'
        if ikey not in ssh_opts:
            ssh_opts += ' -o %s' % ikey
    if git_version < LooseVersion('2.3.0'):
        os.environ['GIT_SSH_OPTS'] = ssh_opts
        wrapper = write_ssh_wrapper(module)
        os.environ['GIT_SSH'] = wrapper
    else:
        full_cmd = os.environ.get('GIT_SSH', os.environ.get('GIT_SSH_COMMAND', 'ssh'))
        if ssh_opts:
            full_cmd += ' ' + ssh_opts
        os.environ['GIT_SSH_COMMAND'] = full_cmd