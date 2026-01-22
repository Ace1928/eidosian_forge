from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def search_resource(config):
    module = config.module
    client = config.client(get_region(module), 'vpc', 'project')
    opts = user_input_parameters(module)
    identity_obj = _build_identity_object(opts)
    query_link = _build_query_link(opts)
    link = build_path(module, 'subnets/{subnet_id}/privateips') + query_link
    result = []
    p = {'marker': ''}
    while True:
        url = link.format(**p)
        r = send_list_request(module, client, url)
        if not r:
            break
        for item in r:
            item = fill_list_resp_body(item)
            if not are_different_dicts(identity_obj, item):
                result.append(item)
        if len(result) > 1:
            break
        p['marker'] = r[-1].get('id')
    return result