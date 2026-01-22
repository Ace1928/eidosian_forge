import errno
import select
from zope.interface import Attribute, Interface, declarations, implementer
from twisted.internet import main, posixbase
from twisted.internet.interfaces import IReactorDaemonize, IReactorFDSet
from twisted.python import failure, log

        Private method called when a FD is ready for reading, writing or was
        lost. Do the work and raise errors where necessary.
        