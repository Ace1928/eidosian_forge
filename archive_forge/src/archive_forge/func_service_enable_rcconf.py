from __future__ import absolute_import, division, print_function
import glob
import json
import os
import platform
import re
import select
import shlex
import subprocess
import tempfile
import time
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.sys_info import get_platform_subclass
from ansible.module_utils.service import fail_if_missing
from ansible.module_utils.six import PY2, b
def service_enable_rcconf(self):
    if self.rcconf_file is None or self.rcconf_key is None or self.rcconf_value is None:
        self.module.fail_json(msg='service_enable_rcconf() requires rcconf_file, rcconf_key and rcconf_value')
    self.changed = None
    entry = '%s="%s"\n' % (self.rcconf_key, self.rcconf_value)
    with open(self.rcconf_file, 'r') as RCFILE:
        new_rc_conf = []
        for rcline in RCFILE:
            rcarray = shlex.split(rcline, comments=True)
            if len(rcarray) >= 1 and '=' in rcarray[0]:
                key, value = rcarray[0].split('=', 1)
                if key == self.rcconf_key:
                    if value.upper() == self.rcconf_value:
                        self.changed = False
                        break
                    else:
                        rcline = entry
                        self.changed = True
            new_rc_conf.append(rcline.strip() + '\n')
    if self.changed is None:
        new_rc_conf.append(entry)
        self.changed = True
    if self.changed is True:
        if self.module.check_mode:
            self.module.exit_json(changed=True, msg='changing service enablement')
        rcconf_dir = os.path.dirname(self.rcconf_file)
        rcconf_base = os.path.basename(self.rcconf_file)
        TMP_RCCONF, tmp_rcconf_file = tempfile.mkstemp(dir=rcconf_dir, prefix='%s-' % rcconf_base)
        for rcline in new_rc_conf:
            os.write(TMP_RCCONF, rcline.encode())
        os.close(TMP_RCCONF)
        self.module.atomic_move(tmp_rcconf_file, self.rcconf_file)