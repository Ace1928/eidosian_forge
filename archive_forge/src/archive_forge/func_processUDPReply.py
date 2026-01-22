import socket, string, types, time, select
import errno
from . import Type,Class,Opcode
from . import Lib
def processUDPReply(self):
    if self.timeout > 0:
        r, w, e = select.select([self.s], [], [], self.timeout)
        if not len(r):
            raise TimeoutError('Timeout')
    self.reply, self.from_address = self.s.recvfrom(65535)
    self.time_finish = time.time()
    self.args['server'] = self.ns
    return self.processReply()