import contextlib
import random
import socket
import sys
import threading
import time
import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse
import dns._ddr
import dns.edns
import dns.exception
import dns.flags
import dns.inet
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.nameserver
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.rdtypes.svcbbase
import dns.reversename
import dns.tsig
def make_resolver_at(where: Union[dns.name.Name, str], port: int=53, family: int=socket.AF_UNSPEC, resolver: Optional[Resolver]=None) -> Resolver:
    """Make a stub resolver using the specified destination as the full resolver.

    *where*, a ``dns.name.Name`` or ``str`` the domain name or IP address of the
    full resolver.

    *port*, an ``int``, the port to use.  If not specified, the default is 53.

    *family*, an ``int``, the address family to use.  This parameter is used if
    *where* is not an address.  The default is ``socket.AF_UNSPEC`` in which case
    the first address returned by ``resolve_name()`` will be used, otherwise the
    first address of the specified family will be used.

    *resolver*, a ``dns.resolver.Resolver`` or ``None``, the resolver to use for
    resolution of hostnames.  If not specified, the default resolver will be used.

    Returns a ``dns.resolver.Resolver`` or raises an exception.
    """
    if resolver is None:
        resolver = get_default_resolver()
    nameservers: List[Union[str, dns.nameserver.Nameserver]] = []
    if isinstance(where, str) and dns.inet.is_address(where):
        nameservers.append(dns.nameserver.Do53Nameserver(where, port))
    else:
        for address in resolver.resolve_name(where, family).addresses():
            nameservers.append(dns.nameserver.Do53Nameserver(address, port))
    res = dns.resolver.Resolver(configure=False)
    res.nameservers = nameservers
    return res