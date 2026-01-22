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
def git_archive(git_path, module, dest, archive, archive_fmt, archive_prefix, version):
    """ Create git archive in given source directory """
    cmd = [git_path, 'archive', '--format', archive_fmt, '--output', archive, version]
    if archive_prefix is not None:
        cmd.insert(-1, '--prefix')
        cmd.insert(-1, archive_prefix)
    rc, out, err = module.run_command(cmd, cwd=dest)
    if rc != 0:
        module.fail_json(msg='Failed to perform archive operation', details='Git archive command failed to create archive %s using %s directory.Error: %s' % (archive, dest, err))
    return (rc, out, err)