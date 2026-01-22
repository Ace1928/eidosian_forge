from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.l3_interfaces.l3_interfaces import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def parse_vifs(self, conf):
    vif_names = re.findall('vif (\\d+)', conf, re.M)
    vifs_list = None
    if vif_names:
        vifs_list = []
        for vif in set(vif_names):
            vif_regex = ' %s .+$' % vif
            cfg = '\n'.join(re.findall(vif_regex, conf, re.M))
            obj = self.parse_attribs(cfg)
            obj['vlan_id'] = vif
            if obj:
                vifs_list.append(obj)
    return vifs_list