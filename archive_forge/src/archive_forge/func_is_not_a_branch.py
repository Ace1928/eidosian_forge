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
def is_not_a_branch(git_path, module, dest):
    branches = get_branches(git_path, module, dest)
    for branch in branches:
        if branch.startswith('* ') and ('no branch' in branch or 'detached from' in branch or 'detached at' in branch):
            return True
    return False