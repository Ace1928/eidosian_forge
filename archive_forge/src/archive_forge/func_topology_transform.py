from __future__ import absolute_import, division, print_function
from ..module_utils.api import NIOS_DTC_POOL
from ..module_utils.api import WapiModule
from ..module_utils.api import normalize_ib_spec
from ansible.module_utils.basic import AnsibleModule
def topology_transform(module):
    topology = module.params['lb_preferred_topology']
    if topology:
        topo_obj = wapi.get_object('dtc:topology', {'name': topology})
        if topo_obj:
            return topo_obj[0]['_ref']
        else:
            module.fail_json(msg='topology %s cannot be found.' % topology)