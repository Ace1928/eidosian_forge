import cherrypy
from cherrypy.test import helper
from cherrypy._json import json
@cherrypy.expose
@json_out
def json_list(self):
    return ['a', 'b', 42]