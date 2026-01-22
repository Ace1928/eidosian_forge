from __future__ import absolute_import, division, print_function
import datetime
import math
import re
import time
import traceback
from collections import namedtuple
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.urls import urlparse
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
@property
def virtual_servers(self):
    result = []
    if self._values['virtual_servers'] is None or 'items' not in self._values['virtual_servers']:
        return result
    for item in self._values['virtual_servers']['items']:
        self._remove_internal_keywords(item, stats=True)
        stats = self._process_vs_stats(item['selfLink'])
        self._remove_internal_keywords(item)
        item['stats'] = stats
        if 'disabled' in item:
            if item['disabled'] in BOOLEANS_TRUE:
                item['disabled'] = flatten_boolean(item['disabled'])
                item['enabled'] = flatten_boolean(not item['disabled'])
        if 'enabled' in item:
            if item['enabled'] in BOOLEANS_TRUE:
                item['enabled'] = flatten_boolean(item['enabled'])
                item['disabled'] = flatten_boolean(not item['enabled'])
        if 'fullPath' in item:
            item['full_path'] = item.pop('fullPath')
        if 'limitMaxBps' in item:
            item['limit_max_bps'] = int(item.pop('limitMaxBps'))
        if 'limitMaxBpsStatus' in item:
            item['limit_max_bps_status'] = item.pop('limitMaxBpsStatus')
        if 'limitMaxConnections' in item:
            item['limit_max_connections'] = int(item.pop('limitMaxConnections'))
        if 'limitMaxConnectionsStatus' in item:
            item['limit_max_connections_status'] = item.pop('limitMaxConnectionsStatus')
        if 'limitMaxPps' in item:
            item['limit_max_pps'] = int(item.pop('limitMaxPps'))
        if 'limitMaxPpsStatus' in item:
            item['limit_max_pps_status'] = item.pop('limitMaxPpsStatus')
        if 'translationAddress' in item:
            item['translation_address'] = item.pop('translationAddress')
        if 'translationPort' in item:
            item['translation_port'] = int(item.pop('translationPort'))
        result.append(item)
    return result