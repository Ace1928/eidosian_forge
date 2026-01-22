import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
def set_store(self, store):
    """Set the Storage for the credential.

        Args:
            store: Storage, an implementation of Storage object.
                   This is needed to store the latest access_token if it
                   has expired and been refreshed. This implementation uses
                   locking to check for updates before updating the
                   access_token.
        """
    self.store = store