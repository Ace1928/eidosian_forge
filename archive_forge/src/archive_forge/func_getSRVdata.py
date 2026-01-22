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
def getSRVdata(self):
    """
        _Service._Proto.Name TTL Class SRV Priority Weight Port Target
        """
    priority = self.get16bit()
    weight = self.get16bit()
    port = self.get16bit()
    target = self.getname()
    return (priority, weight, port, target)