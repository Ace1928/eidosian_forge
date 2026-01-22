from __future__ import absolute_import, division, print_function
import os
import platform
import socket
import traceback
import ansible.module_utils.compat.typing as t
from ansible.module_utils.basic import (
from ansible.module_utils.common.sys_info import get_platform_subclass
from ansible.module_utils.facts.system.service_mgr import ServiceMgrFactCollector
from ansible.module_utils.facts.utils import get_file_lines, get_file_content
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.six import PY3, text_type
def update_permanent_hostname(self):
    name = self.module.params['name']
    all_names = tuple((self.module.run_command([self.scutil, '--get', name_type])[1].strip() for name_type in self.name_types))
    expected_names = tuple((self.scrubbed_name if n == 'LocalHostName' else name for n in self.name_types))
    if all_names != expected_names:
        if not self.module.check_mode:
            self.set_permanent_hostname(name)
        self.changed = True