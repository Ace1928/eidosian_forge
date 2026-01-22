import datetime
from itertools import count
import os
import threading
import time
import urllib.parse
import pytest
import cherrypy
from cherrypy.lib import httputil
from cherrypy.test import helper
@cherrypy.expose
def cacheable(self):
    cherrypy.response.headers['Etag'] = 'bibbitybobbityboo'
    return "Hi, I'm cacheable."