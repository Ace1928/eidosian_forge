import os
import warnings
import incremental
from twisted.application import service, strports
from twisted.internet import interfaces, reactor
from twisted.python import deprecate, reflect, threadpool, usage
from twisted.spread import pb
from twisted.web import demo, distrib, resource, script, server, static, twcgi, wsgi
def opt_add_header(self, header):
    """
        Specify an additional header to be included in all responses. Specified
        as "HeaderName: HeaderValue".
        """
    name, value = header.split(':', 1)
    self['extraHeaders'].append((name.strip(), value.strip()))