from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_create_metadata_full_clone(d, array_index):
    v = navigate_value(d, ['enable_full_clone'], array_index)
    return '0' if v else ''