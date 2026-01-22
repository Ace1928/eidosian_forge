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
def exec_install(self, items, action, pkgs, res):
    cmd = self.yum_basecmd + [action] + pkgs
    if self.releasever:
        cmd.extend(['--releasever=%s' % self.releasever])
    if not self.sslverify:
        cmd.extend(['--setopt', 'sslverify=0'])
    if self.module.check_mode:
        self.module.exit_json(changed=True, results=res['results'], changes=dict(installed=pkgs))
    else:
        res['changes'] = dict(installed=pkgs)
    locale = get_best_parsable_locale(self.module)
    lang_env = dict(LANG=locale, LC_ALL=locale, LC_MESSAGES=locale)
    rc, out, err = self.module.run_command(cmd, environ_update=lang_env)
    if rc == 1:
        for spec in items:
            if '://' in spec and ('No package %s available.' % spec in out or 'Cannot open: %s. Skipping.' % spec in err):
                err = 'Package at %s could not be installed' % spec
                self.module.fail_json(changed=False, msg=err, rc=rc)
    res['rc'] = rc
    res['results'].append(out)
    res['msg'] += err
    res['changed'] = True
    if 'Nothing to do' in out and rc == 0 or 'does not have any packages' in err:
        res['changed'] = False
    if rc != 0:
        res['changed'] = False
        self.module.fail_json(**res)
    if 'No space left on device' in (out or err):
        res['changed'] = False
        res['msg'] = 'No space left on device'
        self.module.fail_json(**res)
    return res