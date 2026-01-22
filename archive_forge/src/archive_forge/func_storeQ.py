import types
import socket
from . import Type
from . import Class
from . import Opcode
from . import Status
import DNS
from .Base import DNSError
from struct import pack as struct_pack
from struct import unpack as struct_unpack
from socket import inet_ntoa, inet_aton, inet_ntop, AF_INET6
def storeQ(self, u):
    q = {}
    q['qname'], q['qtype'], q['qclass'] = u.getQuestion()
    q['qtypestr'] = Type.typestr(q['qtype'])
    q['qclassstr'] = Class.classstr(q['qclass'])
    return q