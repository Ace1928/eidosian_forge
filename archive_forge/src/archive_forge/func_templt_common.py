from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def templt_common(val, tmplt):
    if val.get('facility'):
        tmplt += ' facility {facility}'.format(facility=val['facility'])
    if val.get('severity'):
        tmplt += ' level {level}'.format(level=val['severity'])
    if val.get('protocol'):
        tmplt += ' protocol {protocol}'.format(protocol=val['protocol'])
    return tmplt