import cherrypy
from cherrypy import expose, tools
@expose()
def call_empty(self):
    return 'Mrs. B.J. Smegma'