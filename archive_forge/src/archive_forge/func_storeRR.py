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
def storeRR(self, u):
    r = {}
    r['name'], r['type'], r['class'], r['ttl'], r['rdlength'] = u.getRRheader()
    r['typename'] = Type.typestr(r['type'])
    r['classstr'] = Class.classstr(r['class'])
    mname = 'get%sdata' % r['typename']
    if hasattr(u, mname):
        r['data'] = getattr(u, mname)()
    else:
        r['data'] = u.getbytes(r['rdlength'])
    return r