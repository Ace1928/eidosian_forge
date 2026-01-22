import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
def strict_getattr(module, clsname):
    cls = getattr(module, clsname)
    if issubclass(cls, LiteralProxyClass):
        raise AttributeError(cls)
    else:
        return cls