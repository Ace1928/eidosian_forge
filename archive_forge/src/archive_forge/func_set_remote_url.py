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
def set_remote_url(git_path, module, repo, dest, remote):
    """ updates repo from remote sources """
    remote_url = get_remote_url(git_path, module, dest, remote)
    if remote_url == repo or unfrackgitpath(remote_url) == unfrackgitpath(repo):
        return False
    command = [git_path, 'remote', 'set-url', remote, repo]
    rc, out, err = module.run_command(command, cwd=dest)
    if rc != 0:
        label = 'set a new url %s for %s' % (repo, remote)
        module.fail_json(msg='Failed to %s: %s %s' % (label, out, err))
    return remote_url is not None