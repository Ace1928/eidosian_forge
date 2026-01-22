from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.api import WapiModule
from ..module_utils.api import NIOS_DTC_MONITOR_SNMP
from ..module_utils.api import normalize_ib_spec
def oids(module):
    """ Transform the module argument into a valid WAPI struct
    This function will transform the oids argument into a structure that is a
    valid WAPI structure in the format of:
        {
            comment: <value>,
            condition: <value>,
            first: <value>,
            last: <value>,
            oid: <value>,
            type: <value>,
        }
    It will remove any options that are set to None since WAPI will error on
    that condition.
    The remainder of the value validation is performed by WAPI
    """
    oids = list()
    for item in module.params['oids']:
        oid = dict([(k, v) for k, v in iteritems(item) if v is not None])
        if 'oid' not in oid:
            module.fail_json(msg='oid is required for oid value')
        oids.append(oid)
    return oids