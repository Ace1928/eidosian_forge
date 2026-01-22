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
def get_systemd_service_enabled(self):

    def sysv_exists(name):
        script = '/etc/init.d/' + name
        return os.access(script, os.X_OK)

    def sysv_is_enabled(name):
        return bool(glob.glob('/etc/rc?.d/S??' + name))
    service_name = self.__systemd_unit
    rc, out, err = self.execute_command('%s is-enabled %s' % (self.enable_cmd, service_name))
    if rc == 0:
        return True
    elif out.startswith('disabled'):
        return False
    elif sysv_exists(service_name):
        return sysv_is_enabled(service_name)
    else:
        return False