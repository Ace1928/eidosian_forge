import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.test import helper
@cherrypy.expose
def multipart(self, parts):
    return repr(parts)