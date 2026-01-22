from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def parse_for_packages(stdout):
    packages = []
    data = stdout.split('\n')
    regex = re.compile('^\\(\\d+/\\d+\\)\\s+\\S+\\s+(\\S+)')
    for l in data:
        p = regex.search(l)
        if p:
            packages.append(p.group(1))
    return packages