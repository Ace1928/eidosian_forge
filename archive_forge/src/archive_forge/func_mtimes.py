import os
import sys
import time
import cherrypy
@cherrypy.expose
def mtimes(self):
    return repr(cherrypy.engine.publish('Autoreloader', 'mtimes'))