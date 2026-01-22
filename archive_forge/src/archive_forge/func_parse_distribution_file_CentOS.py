from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def parse_distribution_file_CentOS(self, name, data, path, collected_facts):
    centos_facts = {}
    if 'CentOS Stream' in data:
        centos_facts['distribution_release'] = 'Stream'
        return (True, centos_facts)
    if 'TencentOS Server' in data:
        centos_facts['distribution'] = 'TencentOS'
        return (True, centos_facts)
    return (False, centos_facts)