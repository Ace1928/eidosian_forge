from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def ldap_split_host_port(hostport):
    """
        ldap_split_host_port splits a network address of the form "host:port",
        "host%zone:port", "[host]:port" or "[host%zone]:port" into host or
        host%zone and port.
    """
    result = dict(scheme=None, netlocation=None, host=None, port=None)
    if not hostport:
        return (result, None)
    netlocation = hostport
    scheme_l = '://'
    if '://' in hostport:
        idx = hostport.find(scheme_l)
        result['scheme'] = hostport[:idx]
        netlocation = hostport[idx + len(scheme_l):]
    result['netlocation'] = netlocation
    if netlocation[-1] == ']':
        result['host'] = netlocation
    v = netlocation.rsplit(':', 1)
    if len(v) != 1:
        try:
            result['port'] = int(v[1])
        except ValueError:
            return (None, 'Invalid value specified for port: %s' % v[1])
    result['host'] = v[0]
    return (result, None)