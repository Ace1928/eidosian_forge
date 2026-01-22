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
def resolve_at(where: Union[dns.name.Name, str], qname: Union[dns.name.Name, str], rdtype: Union[dns.rdatatype.RdataType, str]=dns.rdatatype.A, rdclass: Union[dns.rdataclass.RdataClass, str]=dns.rdataclass.IN, tcp: bool=False, source: Optional[str]=None, raise_on_no_answer: bool=True, source_port: int=0, lifetime: Optional[float]=None, search: Optional[bool]=None, port: int=53, family: int=socket.AF_UNSPEC, resolver: Optional[Resolver]=None) -> Answer:
    """Query nameservers to find the answer to the question.

    This is a convenience function that calls ``dns.resolver.make_resolver_at()`` to
    make a resolver, and then uses it to resolve the query.

    See ``dns.resolver.Resolver.resolve`` for more information on the resolution
    parameters, and ``dns.resolver.make_resolver_at`` for information about the resolver
    parameters *where*, *port*, *family*, and *resolver*.

    If making more than one query, it is more efficient to call
    ``dns.resolver.make_resolver_at()`` and then use that resolver for the queries
    instead of calling ``resolve_at()`` multiple times.
    """
    return make_resolver_at(where, port, family, resolver).resolve(qname, rdtype, rdclass, tcp, source, raise_on_no_answer, source_port, lifetime, search)