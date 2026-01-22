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
def next_nameserver(self) -> Tuple[dns.nameserver.Nameserver, bool, float]:
    if self.retry_with_tcp:
        assert self.nameserver is not None
        assert not self.nameserver.is_always_max_size()
        self.tcp_attempt = True
        self.retry_with_tcp = False
        return (self.nameserver, True, 0)
    backoff = 0.0
    if not self.current_nameservers:
        if len(self.nameservers) == 0:
            raise NoNameservers(request=self.request, errors=self.errors)
        self.current_nameservers = self.nameservers[:]
        backoff = self.backoff
        self.backoff = min(self.backoff * 2, 2)
    self.nameserver = self.current_nameservers.pop(0)
    self.tcp_attempt = self.tcp or self.nameserver.is_always_max_size()
    return (self.nameserver, self.tcp_attempt, backoff)