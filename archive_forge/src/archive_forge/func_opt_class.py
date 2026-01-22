import os
import warnings
import incremental
from twisted.application import service, strports
from twisted.internet import interfaces, reactor
from twisted.python import deprecate, reflect, threadpool, usage
from twisted.spread import pb
from twisted.web import demo, distrib, resource, script, server, static, twcgi, wsgi
def opt_class(self, className):
    """
        Create a Resource subclass with a zero-argument constructor.
        """
    classObj = reflect.namedClass(className)
    self['root'] = classObj()