import os
import sys
import time
import cherrypy
def old_term_handler(signum=None, frame=None):
    cherrypy.log('I am an old SIGTERM handler.')
    sys.exit(0)