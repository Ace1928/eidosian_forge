from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_lines
def get_config_state(configfile):
    lines = get_file_lines(configfile, strip=False)
    for line in lines:
        stateline = re.match('^SELINUX=.*$', line)
        if stateline:
            return line.split('=')[1].strip()