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
def callback_application(self):
    """WSGI application for handling the OAuth 2.0 redirect callback.

        If you need finer grained control use `callback_handler` which returns
        just the webapp.RequestHandler.

        Returns:
            A webapp.WSGIApplication that handles the redirect back from the
            server during the OAuth 2.0 dance.
        """
    return webapp.WSGIApplication([(self.callback_path, self.callback_handler())])