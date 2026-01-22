from __future__ import absolute_import, division, print_function
import errno
import filecmp
import grp
import os
import os.path
import platform
import pwd
import shutil
import stat
import tempfile
import traceback
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.six import PY3
def clear_facls(path):
    setfacl = get_bin_path('setfacl')
    acl_command = [setfacl, '-b', path]
    b_acl_command = [to_bytes(x) for x in acl_command]
    locale = get_best_parsable_locale(module)
    rc, out, err = module.run_command(b_acl_command, environ_update=dict(LANG=locale, LC_ALL=locale, LC_MESSAGES=locale))
    if rc != 0:
        raise RuntimeError('Error running "{0}": stdout: "{1}"; stderr: "{2}"'.format(' '.join(b_acl_command), out, err))