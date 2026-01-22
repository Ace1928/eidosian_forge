from __future__ import absolute_import, division, print_function
from functools import partial
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddr_utils import (
from ansible.module_utils._text import to_text
def ipsubnet(value, query='', index=None):
    """Manipulate IPv4/IPv6 subnets"""
    vtype = ipaddr(value, 'type')
    if not vtype:
        return False
    elif vtype == 'address':
        v = ipaddr(value, 'cidr')
    elif vtype == 'network':
        v = ipaddr(value, 'subnet')
    value = netaddr.IPNetwork(v).cidr
    vtype = ipaddr(value, 'type')
    if not query:
        return to_text(value)
    vtotalbits = 128 if value.version == 6 else 32
    if query.isdigit():
        query = int(query)
        if query < 0 or query > vtotalbits:
            return False
        if index is None:
            if vtype == 'address':
                return to_text(value.supernet(query)[0])
            elif vtype == 'network':
                if query - value.prefixlen < 0:
                    msg = 'Requested subnet size of {0} is invalid'.format(to_text(query))
                    raise AnsibleFilterError(msg)
                return to_text(2 ** (query - value.prefixlen))
        index = int(index)
        if vtype == 'address':
            if index > vtotalbits + 1 - index or index < query - vtotalbits:
                return False
            return to_text(value.supernet(query)[index])
        elif vtype == 'network':
            subnets = 2 ** (query - value.prefixlen)
            if index < 0:
                index = index + subnets
            if index < 0 or index >= subnets:
                return False
            return to_text(netaddr.IPNetwork(to_text(netaddr.IPAddress(value.network.value + (index << vtotalbits - query))) + '/' + to_text(query)))
    else:
        vtype = ipaddr(query, 'type')
        if vtype == 'address':
            v = ipaddr(query, 'cidr')
        elif vtype == 'network':
            v = ipaddr(query, 'subnet')
        else:
            msg = 'You must pass a valid subnet or IP address; {0} is invalid'.format(to_text(query))
            raise AnsibleFilterError(msg)
        query = netaddr.IPNetwork(v)
        if value.value >> vtotalbits - query.prefixlen == query.value >> vtotalbits - query.prefixlen:
            return to_text(((value.value & 2 ** (value.prefixlen - query.prefixlen) - 1 << vtotalbits - value.prefixlen) >> vtotalbits - value.prefixlen) + 1)
        msg = '{0} is not in the subnet {1}'.format(value.cidr, query.cidr)
        raise AnsibleFilterError(msg)
    return False