import os
import warnings
import incremental
from twisted.application import service, strports
from twisted.internet import interfaces, reactor
from twisted.python import deprecate, reflect, threadpool, usage
from twisted.spread import pb
from twisted.web import demo, distrib, resource, script, server, static, twcgi, wsgi
def opt_path(self, path):
    """
        <path> is either a specific file or a directory to be set as the root
        of the web server. Use this if you have a directory full of HTML, cgi,
        epy, or rpy files or any other files that you want to be served up raw.
        """
    self['root'] = static.File(os.path.abspath(path))
    self['root'].processors = {'.epy': script.PythonScript, '.rpy': script.ResourceScript}
    self['root'].processors['.cgi'] = twcgi.CGIScript