from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def multi_invoke_attach_data_disk(config, opts, client, timeout):
    module = config.module
    opts1 = opts
    expect = opts['data_volumes']
    current = opts['current_state']['data_volumes']
    if expect and current:
        v = [i['volume_id'] for i in current]
        opts1 = {'data_volumes': [i for i in expect if i['volume_id'] not in v]}
    loop_val = navigate_value(opts1, ['data_volumes'])
    if not loop_val:
        return
    for i in range(len(loop_val)):
        params = build_attach_data_disk_parameters(opts1, {'data_volumes': i})
        r = send_attach_data_disk_request(module, params, client)
        async_wait(config, r, client, timeout)