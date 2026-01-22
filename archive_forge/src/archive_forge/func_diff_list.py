from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def diff_list(want, have):
    adds = [w for w in want if w not in have]
    removes = [h for h in have if h not in want]
    return (adds, removes)