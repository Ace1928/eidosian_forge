from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.logging_global import (
def flatten_facility(self, param):
    temp_param = dict()
    for element, val in iteritems(param):
        if element in ['console', 'global_params', 'syslog']:
            if element != 'syslog' and val.get('facilities'):
                for k, v in iteritems(val.get('facilities')):
                    temp_param[k + element] = {element: {'facilities': v}}
                del val['facilities']
            if val:
                temp_param[element] = {element: val}
        if element in ['files', 'hosts', 'users']:
            for k, v in iteritems(val):
                if v.get('facilities'):
                    for pk, dat in iteritems(v.get('facilities')):
                        temp_param[pk + k] = {element: {'facilities': dat, self.pkey.get(element): v.get(self.pkey.get(element))}}
                    del v['facilities']
                    if len(list(v.keys())) > 1:
                        temp_param[k] = {element: v}
                else:
                    temp_param[k] = {element: v}
    return temp_param