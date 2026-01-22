from __future__ import absolute_import, division, print_function
import copy
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def parse_file_size(line, name, level):
    file_size = None
    match = re.search('logging logfile {0} {1} size (\\S+)'.format(name, level), line, re.M)
    if match:
        file_size = match.group(1)
        if file_size == '8192' or file_size == '4194304':
            file_size = None
    return file_size