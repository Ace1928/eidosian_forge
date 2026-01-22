import sys
from pyside2uic.properties import Properties
from pyside2uic.uiparser import UIParser
from pyside2uic.Compiler import qtproxies
from pyside2uic.Compiler.indenter import createCodeIndenter, getIndenter, \
from pyside2uic.Compiler.qobjectcreator import CompilerCreatorPolicy
from pyside2uic.Compiler.misc import write_import
def createToplevelWidget(self, classname, widgetname):
    indenter = getIndenter()
    indenter.level = 0
    indenter.write('from PySide2 import QtCore, QtGui, QtWidgets')
    indenter.write('')
    indenter.write('class Ui_%s(object):' % self.uiname)
    indenter.indent()
    indenter.write('def setupUi(self, %s):' % widgetname)
    indenter.indent()
    w = self.factory.createQObject(classname, widgetname, (), is_attribute=False, no_instantiation=True)
    w.baseclass = classname
    w.uiclass = 'Ui_%s' % self.uiname
    return w