import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
def setMargin(self, v):
    ProxyClassMember(self, 'setContentsMargins', 0)(v, v, v, v)