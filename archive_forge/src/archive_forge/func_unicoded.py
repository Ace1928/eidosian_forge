import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.test import helper
@cherrypy.expose
@cherrypy.config(**{'tools.encode.on': True})
def unicoded(self):
    return ntou('I am a Ụnicode string.', 'escape')