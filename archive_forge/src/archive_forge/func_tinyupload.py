import os
import cherrypy
from cherrypy.test import helper
@cherrypy.expose
@cherrypy.config(**{'request.body.maxbytes': 100})
def tinyupload(self):
    return cherrypy.request.body.read()