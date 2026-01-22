import cherrypy
from cherrypy.test import helper
@cherrypy.expose
@cherrypy.config(**{'tools.proxy.local': 'X-Host', 'tools.trailing_slash.extra': True})
def xhost(self):
    raise cherrypy.HTTPRedirect('blah')