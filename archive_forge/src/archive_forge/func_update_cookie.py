import sys
import datetime
import os
import time
import threading
import binascii
import pickle
import zc.lockfile
import cherrypy
from cherrypy.lib import httputil
from cherrypy.lib import locking
from cherrypy.lib import is_iterator
def update_cookie(id):
    """Update the cookie every time the session id changes."""
    cherrypy.serving.response.cookie[name] = id