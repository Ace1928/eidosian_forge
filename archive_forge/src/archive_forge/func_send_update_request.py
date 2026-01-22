from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def send_update_request(module, params, client):
    url = build_path(module, 'vpcs/{vpc_id}/subnets/{id}')
    try:
        r = client.put(url, params)
    except HwcClientException as ex:
        msg = 'module(hwc_vpc_subnet): error running api(update), error: %s' % str(ex)
        module.fail_json(msg=msg)
    return r