import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
def objectName(self):
    return self._uic_name.split('.')[-1]