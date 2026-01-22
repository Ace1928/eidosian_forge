import cherrypy
from cherrypy import expose, tools
@expose
def no_call(self):
    return 'Mr E. R. Bradshaw'