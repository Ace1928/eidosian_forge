from the current environment without the need to copy, save and manage
import abc
import copy
import datetime
import io
import json
import re
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import impersonated_credentials
from google.auth import metrics
from google.oauth2 import sts
from google.oauth2 import utils
@_helpers.copy_docstring(credentials.Scoped)
def with_scopes(self, scopes, default_scopes=None):
    kwargs = self._constructor_args()
    kwargs.update(scopes=scopes, default_scopes=default_scopes)
    scoped = self.__class__(**kwargs)
    scoped._metrics_options = self._metrics_options
    return scoped