from __future__ import absolute_import
import logging
import os
import warnings
import six
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import service_account
Proxy to ``self.http``.