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
def unimplemented_error(self):
    system = platform.system()
    distribution = get_distribution()
    if distribution is not None:
        msg_platform = '%s (%s)' % (system, distribution)
    else:
        msg_platform = system
    self.module.fail_json(msg='hostname module cannot be used on platform %s' % msg_platform)