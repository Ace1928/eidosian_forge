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
def dumpM(u):
    print('HEADER:')
    id, qr, opcode, aa, tc, rd, ra, z, rcode, qdcount, ancount, nscount, arcount = u.getHeader()
    print('id=%d,' % id)
    print('qr=%d, opcode=%d, aa=%d, tc=%d, rd=%d, ra=%d, z=%d, rcode=%d,' % (qr, opcode, aa, tc, rd, ra, z, rcode))
    if tc:
        print('*** response truncated! ***')
    if rcode:
        print('*** nonzero error code! (%d) ***' % rcode)
    print('  qdcount=%d, ancount=%d, nscount=%d, arcount=%d' % (qdcount, ancount, nscount, arcount))
    for i in range(qdcount):
        print('QUESTION %d:' % i)
        dumpQ(u)
    for i in range(ancount):
        print('ANSWER %d:' % i)
        dumpRR(u)
    for i in range(nscount):
        print('AUTHORITY RECORD %d:' % i)
        dumpRR(u)
    for i in range(arcount):
        print('ADDITIONAL RECORD %d:' % i)
        dumpRR(u)