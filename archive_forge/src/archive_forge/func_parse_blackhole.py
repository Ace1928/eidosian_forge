from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.static_routes.static_routes import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def parse_blackhole(self, conf):
    blackhole = None
    if conf:
        distance = search('^.*blackhole distance (.\\S+)', conf, M)
        bh = conf.find('blackhole')
        if distance is not None:
            blackhole = {}
            value = distance.group(1).strip("'")
            blackhole['distance'] = int(value)
        elif bh:
            blackhole = {}
            blackhole['type'] = 'blackhole'
    return blackhole