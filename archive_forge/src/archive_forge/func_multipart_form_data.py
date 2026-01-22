import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.test import helper
@cherrypy.expose
def multipart_form_data(self, **kwargs):
    return repr(list(sorted(kwargs.items())))