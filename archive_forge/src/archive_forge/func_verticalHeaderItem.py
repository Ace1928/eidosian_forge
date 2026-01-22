import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
def verticalHeaderItem(self, row):
    return QtWidgets.QTableWidgetItem('%s.verticalHeaderItem(%i)' % (self, row), False, (), noInstantiation=True)