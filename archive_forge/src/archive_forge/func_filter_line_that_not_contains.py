from __future__ import absolute_import, division, print_function
import re
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule
def filter_line_that_not_contains(pattern, content):
    return ''.join([line for line in content.splitlines(True) if not line.contains(pattern)])