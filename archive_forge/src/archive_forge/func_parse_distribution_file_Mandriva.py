from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def parse_distribution_file_Mandriva(self, name, data, path, collected_facts):
    mandriva_facts = {}
    if 'Mandriva' in data:
        mandriva_facts['distribution'] = 'Mandriva'
        version = re.search('DISTRIB_RELEASE="(.*)"', data)
        if version:
            mandriva_facts['distribution_version'] = version.groups()[0]
        release = re.search('DISTRIB_CODENAME="(.*)"', data)
        if release:
            mandriva_facts['distribution_release'] = release.groups()[0]
        mandriva_facts['distribution'] = name
    else:
        return (False, mandriva_facts)
    return (True, mandriva_facts)