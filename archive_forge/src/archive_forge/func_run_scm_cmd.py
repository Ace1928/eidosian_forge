from __future__ import (absolute_import, division, print_function)
import os
import tempfile
from subprocess import Popen, PIPE
import tarfile
import ansible.constants as C
from ansible import context
from ansible.errors import AnsibleError
from ansible.utils.display import Display
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_text, to_native
def run_scm_cmd(cmd, tempdir):
    try:
        stdout = ''
        stderr = ''
        popen = Popen(cmd, cwd=tempdir, stdout=PIPE, stderr=PIPE)
        stdout, stderr = popen.communicate()
    except Exception as e:
        ran = ' '.join(cmd)
        display.debug('ran %s:' % ran)
        raise AnsibleError('when executing %s: %s' % (ran, to_native(e)))
    if popen.returncode != 0:
        raise AnsibleError('- command %s failed in directory %s (rc=%s) - %s' % (' '.join(cmd), tempdir, popen.returncode, to_native(stderr)))