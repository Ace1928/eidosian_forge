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
def get_systemd_status_dict(self):
    rc, out, err = self.execute_command("%s show '%s'" % (self.enable_cmd, self.__systemd_unit))
    if rc != 0:
        self.module.fail_json(msg='failure %d running systemctl show for %r: %s' % (rc, self.__systemd_unit, err))
    elif 'LoadState=not-found' in out:
        self.module.fail_json(msg='systemd could not find the requested service "%r": %s' % (self.__systemd_unit, err))
    key = None
    value_buffer = []
    status_dict = {}
    for line in out.splitlines():
        if '=' in line:
            if not key:
                key, value = line.split('=', 1)
                if value.lstrip().startswith('{'):
                    if value.rstrip().endswith('}'):
                        status_dict[key] = value
                        key = None
                    else:
                        value_buffer.append(value)
                else:
                    status_dict[key] = value
                    key = None
            elif line.rstrip().endswith('}'):
                status_dict[key] = '\n'.join(value_buffer)
                key = None
            else:
                value_buffer.append(value)
        else:
            value_buffer.append(value)
    return status_dict