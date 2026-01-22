from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def cmp_interfaces(self, want, have, tagged):
    result = []
    if tagged:
        tag_key = 'tagged'
    else:
        tag_key = 'untagged'
    if want is None:
        return None
    elif want == '' and have is None:
        return None
    elif want == '' and len(have) > 0:
        pass
    elif not have:
        result = dict(interfaces=[{'name': x, tag_key: True} for x in want])
    elif set(want) != set(have):
        result = dict(interfaces=[{'name': x, tag_key: True} for x in want])
    else:
        return None
    return result