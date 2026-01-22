from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_prefix_limit_payload(prefix_limit):
    pfx_lmt_cfg = {}
    if prefix_limit.get('max_prefixes', None) is not None:
        max_prefixes = prefix_limit['max_prefixes']
        pfx_lmt_cfg.update({'max-prefixes': max_prefixes})
    if prefix_limit.get('prevent_teardown', None) is not None:
        prevent_teardown = prefix_limit['prevent_teardown']
        pfx_lmt_cfg.update({'prevent-teardown': prevent_teardown})
    if prefix_limit.get('warning_threshold', None) is not None:
        warning_threshold = prefix_limit['warning_threshold']
        pfx_lmt_cfg.update({'warning-threshold-pct': warning_threshold})
    if prefix_limit.get('restart_timer', None) is not None:
        restart_timer = prefix_limit['restart_timer']
        pfx_lmt_cfg.update({'restart-timer': restart_timer})
    return pfx_lmt_cfg