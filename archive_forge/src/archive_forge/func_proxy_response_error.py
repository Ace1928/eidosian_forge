import datetime
import hashlib
import http.client as http
import os
import re
import urllib.parse as urlparse
import uuid
from castellan.common import exception as castellan_exception
from castellan import key_manager
import glance_store
from glance_store import location
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
from oslo_utils import encodeutils
from oslo_utils import timeutils as oslo_timeutils
import requests
import webob.exc
from glance.api import common
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import store_utils
from glance.common import timeutils
from glance.common import utils
from glance.common import wsgi
from glance import context as glance_context
import glance.db
import glance.gateway
from glance.i18n import _, _LE, _LI, _LW
import glance.notifier
from glance.quota import keystone as ks_quota
import glance.schema
def proxy_response_error(orig_code, orig_explanation):
    """Construct a webob.exc.HTTPError exception on the fly.

    The webob.exc.HTTPError classes are statically defined, intended
    to be straight subclasses of HTTPError, specifically with *class*
    level definitions of things we need to be dynamic. This method
    returns an exception class instance with those values set
    programmatically so we can raise it to mimic the response we
    got from a remote.
    """

    class ProxiedResponse(webob.exc.HTTPError):
        code = orig_code
        title = orig_explanation
    return ProxiedResponse()