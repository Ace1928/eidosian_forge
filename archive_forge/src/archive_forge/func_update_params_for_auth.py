from __future__ import absolute_import
import os
import re
import json
import mimetypes
import tempfile
from multiprocessing.pool import ThreadPool
from datetime import date, datetime
from six import PY3, integer_types, iteritems, text_type
from six.moves.urllib.parse import quote
from . import models
from .configuration import Configuration
from .rest import ApiException, RESTClientObject
def update_params_for_auth(self, headers, querys, auth_settings):
    """
        Updates header and query params based on authentication setting.

        :param headers: Header parameters dict to be updated.
        :param querys: Query parameters tuple list to be updated.
        :param auth_settings: Authentication setting identifiers list.
        """
    if not auth_settings:
        return
    for auth in auth_settings:
        auth_setting = self.configuration.auth_settings().get(auth)
        if auth_setting:
            if not auth_setting['value']:
                continue
            elif auth_setting['in'] == 'header':
                headers[auth_setting['key']] = auth_setting['value']
            elif auth_setting['in'] == 'query':
                querys.append((auth_setting['key'], auth_setting['value']))
            else:
                raise ValueError('Authentication token must be in `query` or `header`')