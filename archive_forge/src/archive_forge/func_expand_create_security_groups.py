from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_create_security_groups(d, array_index):
    v = d.get('security_groups')
    if not v:
        return None
    return [{'id': i} for i in v]