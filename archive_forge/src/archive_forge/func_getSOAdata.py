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
def getSOAdata(self):
    return (self.getname(), self.getname(), ('serial',) + (self.get32bit(),), ('refresh ',) + prettyTime(self.get32bit()), ('retry',) + prettyTime(self.get32bit()), ('expire',) + prettyTime(self.get32bit()), ('minimum',) + prettyTime(self.get32bit()))