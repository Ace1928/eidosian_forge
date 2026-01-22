from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.api import WapiModule
from ..module_utils.api import NIOS_IPV4_NETWORK, NIOS_IPV6_NETWORK
from ..module_utils.api import NIOS_IPV4_NETWORK_CONTAINER, NIOS_IPV6_NETWORK_CONTAINER
from ..module_utils.api import normalize_ib_spec
from ..module_utils.network import validate_ip_address, validate_ip_v6_address
def check_ip_addr_type(obj_filter, ib_spec):
    """This function will check if the argument ip is type v4/v6 and return appropriate infoblox
       network/networkcontainer type
    """
    ip = obj_filter['network']
    if 'container' in obj_filter and obj_filter['container']:
        check_ip = ip.split('/')
        del ib_spec['container']
        del ib_spec['options']
        if validate_ip_address(check_ip[0]):
            return (NIOS_IPV4_NETWORK_CONTAINER, ib_spec)
        elif validate_ip_v6_address(check_ip[0]):
            return (NIOS_IPV6_NETWORK_CONTAINER, ib_spec)
    else:
        check_ip = ip.split('/')
        del ib_spec['container']
        if validate_ip_address(check_ip[0]):
            return (NIOS_IPV4_NETWORK, ib_spec)
        elif validate_ip_v6_address(check_ip[0]):
            return (NIOS_IPV6_NETWORK, ib_spec)