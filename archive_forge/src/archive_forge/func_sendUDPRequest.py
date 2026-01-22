import socket, string, types, time, select
import errno
from . import Type,Class,Opcode
from . import Lib
def sendUDPRequest(self, server):
    """refactor me"""
    first_socket_error = None
    self.response = None
    for self.ns in server:
        try:
            if self.ns.count(':'):
                if hasattr(socket, 'has_ipv6') and socket.has_ipv6:
                    self.socketInit(socket.AF_INET6, socket.SOCK_DGRAM)
                else:
                    continue
            else:
                self.socketInit(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                self.time_start = time.time()
                self.conn()
                self.s.send(self.request)
                r = self.processUDPReply()
                while r.header['id'] != self.tid or self.from_address[1] != self.port:
                    r = self.processUDPReply()
                self.response = r
            finally:
                self.s.close()
        except socket.error as e:
            first_socket_error = first_socket_error or e
            continue
        except TimeoutError as t:
            first_socket_error = first_socket_error or t
            continue
        if self.response:
            break
    if not self.response and first_socket_error:
        raise first_socket_error