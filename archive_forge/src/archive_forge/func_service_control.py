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
def service_control(self):
    rc, stdout, stderr = self.execute_command('%s -a' % self.lssrc_cmd)
    if rc == 1:
        if stderr:
            self.module.fail_json(msg=stderr)
        else:
            self.module.fail_json(msg=stdout)
    else:
        lines = stdout.splitlines()
        subsystems = []
        groups = []
        for line in lines[1:]:
            subsystem = line.split()[0].strip()
            group = line.split()[1].strip()
            subsystems.append(subsystem)
            if group:
                groups.append(group)
        if self.name in subsystems:
            srccmd_parameter = '-s'
        elif self.name in groups:
            srccmd_parameter = '-g'
    if self.action == 'start':
        srccmd = self.startsrc_cmd
    elif self.action == 'stop':
        srccmd = self.stopsrc_cmd
    elif self.action == 'reload':
        srccmd = self.refresh_cmd
    elif self.action == 'restart':
        self.execute_command('%s %s %s' % (self.stopsrc_cmd, srccmd_parameter, self.name))
        if self.sleep:
            time.sleep(self.sleep)
        srccmd = self.startsrc_cmd
    if self.arguments and self.action in ('start', 'restart'):
        return self.execute_command('%s -a "%s" %s %s' % (srccmd, self.arguments, srccmd_parameter, self.name))
    else:
        return self.execute_command('%s %s %s' % (srccmd, srccmd_parameter, self.name))