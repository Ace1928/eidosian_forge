from __future__ import (absolute_import, division, print_function)
import os
import platform
import re
import ansible.module_utils.compat.typing as t
from ansible.module_utils.common.sys_info import get_distribution, get_distribution_version, \
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector
def parse_distribution_file_OpenWrt(self, name, data, path, collected_facts):
    openwrt_facts = {}
    if 'OpenWrt' not in data:
        return (False, openwrt_facts)
    openwrt_facts['distribution'] = name
    version = re.search('DISTRIB_RELEASE="(.*)"', data)
    if version:
        openwrt_facts['distribution_version'] = version.groups()[0]
    release = re.search('DISTRIB_CODENAME="(.*)"', data)
    if release:
        openwrt_facts['distribution_release'] = release.groups()[0]
    return (True, openwrt_facts)