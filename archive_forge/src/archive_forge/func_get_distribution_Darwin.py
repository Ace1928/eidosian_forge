from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def get_distribution_Darwin(self):
    darwin_facts = {}
    darwin_facts['distribution'] = 'MacOSX'
    rc, out, err = self.module.run_command('/usr/bin/sw_vers -productVersion')
    data = out.split()[-1]
    if data:
        darwin_facts['distribution_major_version'] = data.split('.')[0]
        darwin_facts['distribution_version'] = data
    return darwin_facts