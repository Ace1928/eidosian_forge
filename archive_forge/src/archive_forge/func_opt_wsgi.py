import os
import warnings
import incremental
from twisted.application import service, strports
from twisted.internet import interfaces, reactor
from twisted.python import deprecate, reflect, threadpool, usage
from twisted.spread import pb
from twisted.web import demo, distrib, resource, script, server, static, twcgi, wsgi
def opt_wsgi(self, name):
    """
        The FQPN of a WSGI application object to serve as the root resource of
        the webserver.
        """
    try:
        application = reflect.namedAny(name)
    except (AttributeError, ValueError):
        raise usage.UsageError(f'No such WSGI application: {name!r}')
    pool = threadpool.ThreadPool()
    reactor.callWhenRunning(pool.start)
    reactor.addSystemEventTrigger('after', 'shutdown', pool.stop)
    self['root'] = wsgi.WSGIResource(reactor, pool, application)