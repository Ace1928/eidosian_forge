import os.path
import cherrypy
@cherrypy.expose
def greetUser(self, name=None):
    if name:
        return "Hey %s, what's up?" % name
    elif name is None:
        return 'Please enter your name <a href="./">here</a>.'
    else:
        return 'No, really, enter your name <a href="./">here</a>.'