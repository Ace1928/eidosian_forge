from __future__ import absolute_import, division, print_function
import re
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule
def filter_line_that_contains_ipv6(content):
    return filter_line_that_match_func(ipv6_regexp.search, content)