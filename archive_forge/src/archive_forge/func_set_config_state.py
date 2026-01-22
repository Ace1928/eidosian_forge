from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_lines
def set_config_state(module, state, configfile):
    stateline = 'SELINUX=%s' % state
    lines = get_file_lines(configfile, strip=False)
    tmpfd, tmpfile = tempfile.mkstemp()
    with open(tmpfile, 'w') as write_file:
        line_found = False
        for line in lines:
            if re.match('^SELINUX=.*$', line):
                line_found = True
            write_file.write(re.sub('^SELINUX=.*', stateline, line) + '\n')
        if not line_found:
            write_file.write('SELINUX=%s\n' % state)
    module.atomic_move(tmpfile, configfile)