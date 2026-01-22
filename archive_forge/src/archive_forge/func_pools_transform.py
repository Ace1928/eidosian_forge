from __future__ import absolute_import, division, print_function
from ..module_utils.api import NIOS_DTC_LBDN
from ..module_utils.api import WapiModule
from ..module_utils.api import normalize_ib_spec
from ansible.module_utils.basic import AnsibleModule
def pools_transform(module):
    pool_list = list()
    if module.params['pools']:
        for pool in module.params['pools']:
            pool_obj = wapi.get_object('dtc:pool', {'name': pool['pool']})
            if 'ratio' not in pool:
                pool['ratio'] = 1
            if pool_obj:
                pool_list.append({'pool': pool_obj[0]['_ref'], 'ratio': pool['ratio']})
            else:
                module.fail_json(msg='pool %s cannot be found.' % pool)
    return pool_list