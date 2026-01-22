import cherrypy
from cherrypy.test import helper
def augment_params():
    cherrypy.request.params['test'] = 'test'