from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.api import NIOS_IPV4_FIXED_ADDRESS, NIOS_IPV6_FIXED_ADDRESS
from ..module_utils.api import WapiModule
from ..module_utils.api import normalize_ib_spec
from ..module_utils.network import validate_ip_address, validate_ip_v6_address
def validate_ip_addr_type(ip, arg_spec, module):
    """This function will check if the argument ip is type v4/v6 and return appropriate infoblox network type
    """
    check_ip = ip.split('/')
    if validate_ip_address(check_ip[0]) and 'ipaddr' in arg_spec:
        arg_spec['ipv4addr'] = arg_spec.pop('ipaddr')
        module.params['ipv4addr'] = module.params.pop('ipaddr')
        del arg_spec['duid']
        del module.params['duid']
        if module.params['mac'] is None:
            raise ValueError("the 'mac' address of the object must be specified")
        module.params['mac'] = module.params['mac'].lower()
        return (NIOS_IPV4_FIXED_ADDRESS, arg_spec, module)
    elif validate_ip_v6_address(check_ip[0]) and 'ipaddr' in arg_spec:
        arg_spec['ipv6addr'] = arg_spec.pop('ipaddr')
        module.params['ipv6addr'] = module.params.pop('ipaddr')
        del arg_spec['mac']
        del module.params['mac']
        if module.params['duid'] is None:
            raise ValueError("the 'duid' of the object must be specified")
        module.params['duid'] = module.params['duid'].lower()
        return (NIOS_IPV6_FIXED_ADDRESS, arg_spec, module)