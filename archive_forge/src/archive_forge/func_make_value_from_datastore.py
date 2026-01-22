import cgi
import json
import logging
import os
import pickle
import threading
from google.appengine.api import app_identity
from google.appengine.api import memcache
from google.appengine.api import users
from google.appengine.ext import db
from google.appengine.ext.webapp.util import login_required
import webapp2 as webapp
import oauth2client
from oauth2client import _helpers
from oauth2client import client
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import xsrfutil
def make_value_from_datastore(self, value):
    logger.info('make: Got type ' + str(type(value)))
    if value is None:
        return None
    if len(value) == 0:
        return None
    try:
        credentials = client.Credentials.new_from_json(value)
    except ValueError:
        credentials = None
    return credentials