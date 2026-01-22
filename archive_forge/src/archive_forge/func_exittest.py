import os
import signal
import time
from http.client import BadStatusLine
import pytest
import portend
import cherrypy
import cherrypy.process.servers
from cherrypy.test import helper
def exittest():
    self.getPage('/')
    self.assertBody('Hello World')
    engine.exit()