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
def get_value_for_datastore(self, model_instance):
    logger.info('get: Got type ' + str(type(model_instance)))
    cred = super(CredentialsProperty, self).get_value_for_datastore(model_instance)
    if cred is None:
        cred = ''
    else:
        cred = cred.to_json()
    return db.Blob(cred)