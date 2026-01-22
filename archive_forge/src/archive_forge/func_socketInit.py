import socket, string, types, time, select
import errno
from . import Type,Class,Opcode
from . import Lib
def socketInit(self, a, b):
    self.s = socket.socket(a, b)