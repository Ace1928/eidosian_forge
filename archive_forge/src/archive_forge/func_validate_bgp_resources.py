from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def validate_bgp_resources(module, want, have, check_neighbors=None):
    undefined_resources = get_undefined_bgps(want, have, check_neighbors)
    if undefined_resources:
        err = 'Resource not found! {res}'.format(res=undefined_resources)
        module.fail_json(msg=err, code=404)