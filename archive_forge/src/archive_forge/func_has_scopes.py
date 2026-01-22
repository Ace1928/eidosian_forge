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
def has_scopes(self, scopes):
    """Verify that the credentials are authorized for the given scopes.

        Returns True if the credentials authorized scopes contain all of the
        scopes given.

        Args:
            scopes: list or string, the scopes to check.

        Notes:
            There are cases where the credentials are unaware of which scopes
            are authorized. Notably, credentials obtained and stored before
            this code was added will not have scopes, AccessTokenCredentials do
            not have scopes. In both cases, you can use refresh_scopes() to
            obtain the canonical set of scopes.
        """
    scopes = _helpers.string_to_scopes(scopes)
    return set(scopes).issubset(self.scopes)