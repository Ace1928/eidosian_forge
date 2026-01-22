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
def sign_blob(self, blob):
    """Cryptographically sign a blob (of bytes).

        Implements abstract method
        :meth:`oauth2client.client.AssertionCredentials.sign_blob`.

        Args:
            blob: bytes, Message to be signed.

        Returns:
            tuple, A pair of the private key ID used to sign the blob and
            the signed contents.
        """
    return app_identity.sign_blob(blob)