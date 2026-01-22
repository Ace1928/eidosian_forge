import os.path
import cherrypy
@cherrypy.expose
def show_msg(self):
    return 'Hello world!'