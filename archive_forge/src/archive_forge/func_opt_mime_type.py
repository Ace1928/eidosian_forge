import os
import warnings
import incremental
from twisted.application import service, strports
from twisted.internet import interfaces, reactor
from twisted.python import deprecate, reflect, threadpool, usage
from twisted.spread import pb
from twisted.web import demo, distrib, resource, script, server, static, twcgi, wsgi
def opt_mime_type(self, defaultType):
    """
        Specify the default mime-type for static files.
        """
    if not isinstance(self['root'], static.File):
        raise usage.UsageError('You can only use --mime_type after --path.')
    self['root'].defaultType = defaultType