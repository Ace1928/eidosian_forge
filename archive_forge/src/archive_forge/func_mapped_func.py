import sys
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy._cptree import Application
from cherrypy.test import helper
@cherrypy.expose
def mapped_func(self, ID=None):
    return 'ID is %s' % ID