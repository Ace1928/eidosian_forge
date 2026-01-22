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
def ssh_supports_acceptnewhostkey(module):
    try:
        ssh_path = get_bin_path('ssh')
    except ValueError as err:
        module.fail_json(msg='Remote host is missing ssh command, so you cannot use acceptnewhostkey option.', details=to_text(err))
    supports_acceptnewhostkey = True
    cmd = [ssh_path, '-o', 'StrictHostKeyChecking=accept-new', '-V']
    rc, stdout, stderr = module.run_command(cmd)
    if rc != 0:
        supports_acceptnewhostkey = False
    return supports_acceptnewhostkey