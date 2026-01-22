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
def query_result(self, response: Optional[dns.message.Message], ex: Optional[Exception]) -> Tuple[Optional[Answer], bool]:
    assert self.nameserver is not None
    if ex:
        assert response is None
        self.errors.append((str(self.nameserver), self.tcp_attempt, self.nameserver.answer_port(), ex, response))
        if isinstance(ex, dns.exception.FormError) or isinstance(ex, EOFError) or isinstance(ex, OSError) or isinstance(ex, NotImplementedError):
            self.nameservers.remove(self.nameserver)
        elif isinstance(ex, dns.message.Truncated):
            if self.tcp_attempt:
                self.nameservers.remove(self.nameserver)
            else:
                self.retry_with_tcp = True
        return (None, False)
    assert response is not None
    assert isinstance(response, dns.message.QueryMessage)
    rcode = response.rcode()
    if rcode == dns.rcode.NOERROR:
        try:
            answer = Answer(self.qname, self.rdtype, self.rdclass, response, self.nameserver.answer_nameserver(), self.nameserver.answer_port())
        except Exception as e:
            self.errors.append((str(self.nameserver), self.tcp_attempt, self.nameserver.answer_port(), e, response))
            self.nameservers.remove(self.nameserver)
            return (None, False)
        if self.resolver.cache:
            self.resolver.cache.put((self.qname, self.rdtype, self.rdclass), answer)
        if answer.rrset is None and self.raise_on_no_answer:
            raise NoAnswer(response=answer.response)
        return (answer, True)
    elif rcode == dns.rcode.NXDOMAIN:
        try:
            answer = Answer(self.qname, dns.rdatatype.ANY, dns.rdataclass.IN, response)
        except Exception as e:
            self.errors.append((str(self.nameserver), self.tcp_attempt, self.nameserver.answer_port(), e, response))
            self.nameservers.remove(self.nameserver)
            return (None, False)
        self.nxdomain_responses[self.qname] = response
        if self.resolver.cache:
            self.resolver.cache.put((self.qname, dns.rdatatype.ANY, self.rdclass), answer)
        return (None, True)
    elif rcode == dns.rcode.YXDOMAIN:
        yex = YXDOMAIN()
        self.errors.append((str(self.nameserver), self.tcp_attempt, self.nameserver.answer_port(), yex, response))
        raise yex
    else:
        if rcode != dns.rcode.SERVFAIL or not self.resolver.retry_servfail:
            self.nameservers.remove(self.nameserver)
        self.errors.append((str(self.nameserver), self.tcp_attempt, self.nameserver.answer_port(), dns.rcode.to_text(rcode), response))
        return (None, False)