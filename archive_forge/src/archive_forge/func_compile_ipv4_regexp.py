from __future__ import absolute_import, division, print_function
import re
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule
def compile_ipv4_regexp():
    r = '((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}'
    r += '(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])'
    return re.compile(r)