from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def build_create_parameters(opts):
    params = dict()
    v = navigate_value(opts, ['ip_address'], None)
    if not is_empty_value(v):
        params['ip_address'] = v
    v = navigate_value(opts, ['subnet_id'], None)
    if not is_empty_value(v):
        params['subnet_id'] = v
    if not params:
        return params
    params = {'privateips': [params]}
    return params