from __future__ import print_function
import sys
import traceback
import pdb
import inspect
from .core import Construct, Subconstruct
from .lib import HexString, Container, ListContainer
def handle_exc(self, msg=None):
    print('=' * 80)
    print('Debugging exception of %s:' % (self.subcon,))
    print(''.join(traceback.format_exception(*sys.exc_info())[1:]))
    if msg:
        print(msg)
    pdb.post_mortem(sys.exc_info()[2])
    print('=' * 80)