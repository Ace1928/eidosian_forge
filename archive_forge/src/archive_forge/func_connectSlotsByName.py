import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
def connectSlotsByName(cls, *args):
    ProxyClassMember(cls, 'connectSlotsByName', 0)(*args)