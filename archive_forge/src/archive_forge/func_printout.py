from __future__ import print_function
import sys
import traceback
import pdb
import inspect
from .core import Construct, Subconstruct
from .lib import HexString, Container, ListContainer
def printout(self, stream, context):
    obj = Container()
    if self.show_stream:
        obj.stream_position = stream.tell()
        follows = stream.read(self.stream_lookahead)
        if not follows:
            obj.following_stream_data = 'EOF reached'
        else:
            stream.seek(-len(follows), 1)
            obj.following_stream_data = HexString(follows)
        print
    if self.show_context:
        obj.context = context
    if self.show_stack:
        obj.stack = ListContainer()
        frames = [s[0] for s in inspect.stack()][1:-1]
        frames.reverse()
        for f in frames:
            a = Container()
            a.__update__(f.f_locals)
            obj.stack.append(a)
    print('=' * 80)
    print('Probe', self.printname)
    print(obj)
    print('=' * 80)