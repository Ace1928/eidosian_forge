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
def get_submodule_versions(git_path, module, dest, version='HEAD'):
    cmd = [git_path, 'submodule', 'foreach', git_path, 'rev-parse', version]
    rc, out, err = module.run_command(cmd, cwd=dest)
    if rc != 0:
        module.fail_json(msg='Unable to determine hashes of submodules', stdout=out, stderr=err, rc=rc)
    submodules = {}
    subm_name = None
    for line in out.splitlines():
        if line.startswith("Entering '"):
            subm_name = line[10:-1]
        elif len(line.strip()) == 40:
            if subm_name is None:
                module.fail_json()
            submodules[subm_name] = line.strip()
            subm_name = None
        else:
            module.fail_json(msg='Unable to parse submodule hash line: %s' % line.strip())
    if subm_name is not None:
        module.fail_json(msg='Unable to find hash for submodule: %s' % subm_name)
    return submodules