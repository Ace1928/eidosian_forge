from __future__ import absolute_import, division, print_function
import argparse
import os
import re
import sys
import tempfile
import operator
import shlex
import traceback
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule, is_executable, missing_required_lib
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.six import PY3
def setup_virtualenv(module, env, chdir, out, err):
    if module.check_mode:
        module.exit_json(changed=True)
    cmd = shlex.split(module.params['virtualenv_command'])
    if os.path.basename(cmd[0]) == cmd[0]:
        cmd[0] = module.get_bin_path(cmd[0], True)
    if module.params['virtualenv_site_packages']:
        cmd.append('--system-site-packages')
    else:
        cmd_opts = _get_cmd_options(module, cmd[0])
        if '--no-site-packages' in cmd_opts:
            cmd.append('--no-site-packages')
    virtualenv_python = module.params['virtualenv_python']
    if not _is_venv_command(module.params['virtualenv_command']):
        if virtualenv_python:
            cmd.append('-p%s' % virtualenv_python)
        elif PY3:
            cmd.append('-p%s' % sys.executable)
    elif module.params['virtualenv_python']:
        module.fail_json(msg='virtualenv_python should not be used when using the venv module or pyvenv as virtualenv_command')
    cmd.append(env)
    rc, out_venv, err_venv = module.run_command(cmd, cwd=chdir)
    out += out_venv
    err += err_venv
    if rc != 0:
        _fail(module, cmd, out, err)
    return (out, err)