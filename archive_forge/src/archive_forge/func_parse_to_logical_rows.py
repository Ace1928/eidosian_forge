from __future__ import absolute_import, division, print_function
import re
import time
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import load_config, run_commands
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import debugOutput, check_args
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import cnos_argument_spec
from ansible.module_utils._text import to_text
def parse_to_logical_rows(out):
    relevant_data = False
    cur_row = []
    for line in out.splitlines():
        if not line:
            'Skip empty lines.'
            continue
        if '0' < line[0] <= '9':
            'Line starting with a number.'
            if len(cur_row) > 0:
                yield cur_row
                cur_row = []
            relevant_data = True
        if relevant_data:
            data = line.strip().split('(')
            cur_row.append(data[0])
    yield cur_row