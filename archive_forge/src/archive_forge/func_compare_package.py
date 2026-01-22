from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import shlex_quote
def compare_package(version1, version2):
    """ Compare version packages.
        Return values:
        -1 first minor
        0 equal
        1 first greater """

    def normalize(v):
        return [int(x) for x in re.sub('(\\.0+)*$', '', v).split('.')]
    normalized_version1 = normalize(version1)
    normalized_version2 = normalize(version2)
    if normalized_version1 == normalized_version2:
        rc = 0
    elif normalized_version1 < normalized_version2:
        rc = -1
    else:
        rc = 1
    return rc