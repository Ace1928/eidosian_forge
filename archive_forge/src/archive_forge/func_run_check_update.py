from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, respawn_module
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
import errno
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from ansible.module_utils.urls import fetch_file
def run_check_update(self):
    if self.releasever:
        rc, out, err = self.module.run_command(self.yum_basecmd + ['check-update'] + ['--releasever=%s' % self.releasever])
    else:
        rc, out, err = self.module.run_command(self.yum_basecmd + ['check-update'])
    return (rc, out, err)