import os
import re
import cherrypy
from cherrypy.process import servers
from cherrypy.test import helper
def erase_script_name(environ, start_response):
    environ['SCRIPT_NAME'] = ''
    return cherrypy.tree(environ, start_response)