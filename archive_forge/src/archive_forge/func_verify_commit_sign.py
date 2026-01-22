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
def verify_commit_sign(git_path, module, dest, version, gpg_whitelist):
    if version in get_annotated_tags(git_path, module, dest):
        git_sub = 'verify-tag'
    else:
        git_sub = 'verify-commit'
    cmd = '%s %s %s' % (git_path, git_sub, version)
    if gpg_whitelist:
        cmd += ' --raw'
    rc, out, err = module.run_command(cmd, cwd=dest)
    if rc != 0:
        module.fail_json(msg='Failed to verify GPG signature of commit/tag "%s"' % version, stdout=out, stderr=err, rc=rc)
    if gpg_whitelist:
        fingerprint = get_gpg_fingerprint(err)
        if fingerprint not in gpg_whitelist:
            module.fail_json(msg='The gpg_whitelist does not include the public key "%s" for this commit' % fingerprint, stdout=out, stderr=err, rc=rc)
    return (rc, out, err)