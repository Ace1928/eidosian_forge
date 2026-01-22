import base64
import boto
import boto.auth_handler
import boto.exception
import boto.plugin
import boto.utils
import copy
import datetime
from email.utils import formatdate
import hmac
import os
import posixpath
from boto.compat import urllib, encodebytes, parse_qs_safe, urlparse, six
from boto.auth_handler import AuthHandler
from boto.exception import BotoClientError
from boto.utils import get_utf8able_str
def mangle_path_and_params(self, req):
    """
        Returns a copy of the request object with fixed ``auth_path/params``
        attributes from the original.
        """
    modified_req = copy.copy(req)
    parsed_path = urllib.parse.urlparse(modified_req.auth_path)
    modified_req.auth_path = parsed_path.path
    if modified_req.params is None:
        modified_req.params = {}
    else:
        copy_params = req.params.copy()
        modified_req.params = copy_params
    raw_qs = parsed_path.query
    existing_qs = parse_qs_safe(raw_qs, keep_blank_values=True)
    for key, value in existing_qs.items():
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                existing_qs[key] = value[0]
    modified_req.params.update(existing_qs)
    return modified_req