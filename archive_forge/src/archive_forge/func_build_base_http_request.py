from datetime import datetime
import errno
import os
import random
import re
import socket
import sys
import time
import xml.sax
import copy
from boto import auth
from boto import auth_handler
import boto
import boto.utils
import boto.handler
import boto.cacerts
from boto import config, UserAgent
from boto.compat import six, http_client, urlparse, quote, encodebytes
from boto.exception import AWSConnectionError
from boto.exception import BotoClientError
from boto.exception import BotoServerError
from boto.exception import PleaseRetryException
from boto.exception import S3ResponseError
from boto.provider import Provider
from boto.resultset import ResultSet
def build_base_http_request(self, method, path, auth_path, params=None, headers=None, data='', host=None):
    path = self.get_path(path)
    if auth_path is not None:
        auth_path = self.get_path(auth_path)
    if params is None:
        params = {}
    else:
        params = params.copy()
    if headers is None:
        headers = {}
    else:
        headers = headers.copy()
    if self.host_header and (not boto.utils.find_matching_headers('host', headers)):
        headers['host'] = self.host_header
    host = host or self.host
    if self.use_proxy and (not self.skip_proxy(host)):
        if not auth_path:
            auth_path = path
        path = self.prefix_proxy_to_path(path, host)
        if self.proxy_user and self.proxy_pass and (not self.is_secure):
            headers.update(self.get_proxy_auth_header())
    return HTTPRequest(method, self.protocol, host, self.port, path, auth_path, params, headers, data)