import abc
import collections
import functools
import re
import uuid
import wsgiref.util
import flask
from flask import blueprints
import flask_restful
import flask_restful.utils
import http.client
from oslo_log import log
from oslo_log import versionutils
from oslo_serialization import jsonutils
from keystone.common import authorization
from keystone.common import context
from keystone.common import driver_hints
from keystone.common import json_home
from keystone.common.rbac_enforcer import enforcer
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
@staticmethod
def query_filter_is_true(filter_name):
    """Determine if bool query param is 'True'.

        We treat this the same way as we do for policy
        enforcement:

        {bool_param}=0 is treated as False

        Any other value is considered to be equivalent to
        True, including the absence of a value (but existence
        as a parameter).

        False Examples for param named `p`:

           * http://host/url
           * http://host/url?p=0

        All other forms of the param 'p' would be result in a True value
        including: `http://host/url?param`.
        """
    val = False
    if filter_name in flask.request.args:
        filter_value = flask.request.args.get(filter_name)
        if isinstance(filter_value, str) and filter_value == '0':
            val = False
        else:
            val = True
    return val