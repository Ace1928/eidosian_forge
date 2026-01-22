from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def fill_list_resp_volume_image_metadata(value):
    if not value:
        return None
    result = dict()
    result['id'] = value.get('id')
    return result