from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer
from twisted.persisted import sob
from twisted.plugin import IPlugin
from twisted.python import components
from twisted.python.reflect import namedAny
def loadApplication(filename, kind, passphrase=None):
    """
    Load Application from a given file.

    The serialization format it was saved in should be given as
    C{kind}, and is one of C{pickle}, C{source}, C{xml} or C{python}. If
    C{passphrase} is given, the application was encrypted with the
    given passphrase.

    @type filename: C{str}
    @type kind: C{str}
    @type passphrase: C{str}
    """
    if kind == 'python':
        application = sob.loadValueFromFile(filename, 'application')
    else:
        application = sob.load(filename, kind)
    return application