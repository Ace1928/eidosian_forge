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
def check_oauth(request_handler, *args, **kwargs):
    if self._in_error:
        self._display_error_message(request_handler)
        return
    user = users.get_current_user()
    if not user:
        request_handler.redirect(users.create_login_url(request_handler.request.uri))
        return
    self._create_flow(request_handler)
    self.flow.params['state'] = _build_state_value(request_handler, user)
    self.credentials = self._storage_class(self._credentials_class, None, self._credentials_property_name, user=user).get()
    if not self.has_credentials():
        return request_handler.redirect(self.authorize_url())
    try:
        resp = method(request_handler, *args, **kwargs)
    except client.AccessTokenRefreshError:
        return request_handler.redirect(self.authorize_url())
    finally:
        self.credentials = None
    return resp