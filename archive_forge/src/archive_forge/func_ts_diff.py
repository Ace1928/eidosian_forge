from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from urllib.parse import quote
def ts_diff(want, have):
    htss = {}
    wtss = {}
    dtss = []
    for hts in have.get('trust_stores') or []:
        htss[hts.get('name')] = hts
    for wts in want.get('trust_stores') or []:
        wtss[wts.get('name')] = wts
    for tsn, ts in wtss.items():
        dts = dict(htss.get(tsn))
        for k, v in ts.items():
            if not isinstance(dts.get(k), list) and (not isinstance(dts.get(k), dict)):
                if dts.get(k) != v:
                    dts[k] = v
            elif v is not None:
                dts[k] = v
        if dts != htss.get(tsn):
            dtss.append(dts)
    return dtss