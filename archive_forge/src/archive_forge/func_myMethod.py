import sys
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy._cptree import Application
from cherrypy.test import helper
@cherrypy.expose
@cherrypy.config(**{'tools.trailing_slash.extra': True})
def myMethod(self):
    return 'myMethod from dir1, path_info is:' + repr(cherrypy.request.path_info)