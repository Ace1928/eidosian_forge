from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_modify_prefix_request(self, prefix, conf_afi):
    """Create a REST API request to update/merge/create  the prefix specified by the
        "prefix" input parameter."""
    pfx_payload = {}
    prefix_val = prefix.get('prefix', None)
    sequence = prefix.get('sequence', None)
    action = prefix.get('action', None)
    if not prefix_val or not sequence or (not action):
        return None
    prefix_net = self.set_ipaddress_net_attrs(prefix_val, conf_afi)
    ge = prefix.get('ge', None)
    le = prefix.get('le', None)
    pfx_payload['ip-prefix'] = prefix_val
    pfx_payload['sequence-number'] = sequence
    masklength_range_str = self.get_masklength_range_string(ge, le, prefix_net)
    pfx_payload['masklength-range'] = masklength_range_str
    pfx_config = {}
    pfx_config['sequence-number'] = sequence
    pfx_config['ip-prefix'] = prefix_val
    pfx_config['masklength-range'] = pfx_payload['masklength-range']
    pfx_config['openconfig-routing-policy-ext:action'] = action.upper()
    pfx_payload['config'] = pfx_config
    return pfx_payload