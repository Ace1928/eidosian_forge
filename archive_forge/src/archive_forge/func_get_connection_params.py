import copy
from http import client as http_client
import io
import logging
import os
import socket
import ssl
from urllib import parse as urlparse
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
from oslo_utils import importutils
from magnumclient import exceptions
@staticmethod
def get_connection_params(endpoint, **kwargs):
    parts = urlparse.urlparse(endpoint)
    path = parts.path
    path = path.rstrip('/').rstrip(API_VERSION)
    _args = (parts.hostname, parts.port, path)
    _kwargs = {'timeout': float(kwargs.get('timeout')) if kwargs.get('timeout') else 600}
    if parts.scheme == 'https':
        _class = VerifiedHTTPSConnection
        _kwargs['ca_file'] = kwargs.get('ca_file', None)
        _kwargs['cert_file'] = kwargs.get('cert_file', None)
        _kwargs['key_file'] = kwargs.get('key_file', None)
        _kwargs['insecure'] = kwargs.get('insecure', False)
    elif parts.scheme == 'http':
        _class = http_client.HTTPConnection
    else:
        msg = 'Unsupported scheme: %s' % parts.scheme
        raise exceptions.EndpointException(msg)
    return (_class, _args, _kwargs)