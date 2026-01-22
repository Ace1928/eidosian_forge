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
def get_systemd_service_status(self):
    d = self.get_systemd_status_dict()
    if d.get('ActiveState') == 'active':
        self.running = True
        self.crashed = False
    elif d.get('ActiveState') == 'failed':
        self.running = False
        self.crashed = True
    elif d.get('ActiveState') is None:
        self.module.fail_json(msg='No ActiveState value in systemctl show output for %r' % (self.__systemd_unit,))
    else:
        self.running = False
        self.crashed = False
    return self.running