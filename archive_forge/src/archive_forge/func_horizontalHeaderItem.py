import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
def horizontalHeaderItem(self, col):
    return QtWidgets.QTableWidgetItem('%s.horizontalHeaderItem(%i)' % (self, col), False, (), noInstantiation=True)