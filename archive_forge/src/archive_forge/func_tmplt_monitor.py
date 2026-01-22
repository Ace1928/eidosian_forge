from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_monitor(config_data):
    return tmplt_common(config_data.get('monitor'), 'logging monitor')