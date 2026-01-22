from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def preprocess_mgmt_vrf_for_overridden(self, want, have):
    new_want = deepcopy(want)
    new_have = deepcopy(have)
    h_conf = next((vrf for vrf in new_have if vrf['name'] == MGMT_VRF_NAME), None)
    if h_conf:
        conf = next((vrf for vrf in new_want if vrf['name'] == MGMT_VRF_NAME), None)
        if conf:
            mv_intfs = []
            if conf.get('members', None) and conf['members'].get('interfaces', None):
                mv_intfs = conf['members'].get('interfaces', [])
            h_mv_intfs = []
            if h_conf.get('members', None) and h_conf['members'].get('interfaces', None):
                h_mv_intfs = h_conf['members'].get('interfaces', [])
            mv_intfs.sort(key=lambda x: x['name'])
            h_mv_intfs.sort(key=lambda x: x['name'])
            if mv_intfs == h_mv_intfs:
                new_want.remove(conf)
                new_have.remove(h_conf)
            elif not h_mv_intfs:
                new_have.remove(h_conf)
        else:
            new_have.remove(h_conf)
    return (new_want, new_have)