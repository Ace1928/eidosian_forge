from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def set_readonly_options(opts, states):
    opts['config_drive'] = states.get('config_drive')
    opts['created'] = states.get('created')
    opts['disk_config_type'] = states.get('disk_config_type')
    opts['host_name'] = states.get('host_name')
    opts['image_name'] = states.get('image_name')
    set_readonly_nics(opts.get('nics'), states.get('nics'))
    opts['power_state'] = states.get('power_state')
    set_readonly_root_volume(opts.get('root_volume'), states.get('root_volume'))
    opts['server_alias'] = states.get('server_alias')
    opts['status'] = states.get('status')