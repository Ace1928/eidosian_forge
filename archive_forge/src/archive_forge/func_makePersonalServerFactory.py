import os
import warnings
import incremental
from twisted.application import service, strports
from twisted.internet import interfaces, reactor
from twisted.python import deprecate, reflect, threadpool, usage
from twisted.spread import pb
from twisted.web import demo, distrib, resource, script, server, static, twcgi, wsgi
def makePersonalServerFactory(site):
    """
    Create and return a factory which will respond to I{distrib} requests
    against the given site.

    @type site: L{twisted.web.server.Site}
    @rtype: L{twisted.internet.protocol.Factory}
    """
    return pb.PBServerFactory(distrib.ResourcePublisher(site))