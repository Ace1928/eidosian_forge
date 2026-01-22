import os
import warnings
import incremental
from twisted.application import service, strports
from twisted.internet import interfaces, reactor
from twisted.python import deprecate, reflect, threadpool, usage
from twisted.spread import pb
from twisted.web import demo, distrib, resource, script, server, static, twcgi, wsgi
def opt_https(self, port):
    """
        (DEPRECATED: use --listen)
        Port to listen on for Secure HTTP.
        """
    msg = deprecate.getDeprecationWarningString(self.opt_https, incremental.Version('Twisted', 18, 4, 0))
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
    self['https'] = port