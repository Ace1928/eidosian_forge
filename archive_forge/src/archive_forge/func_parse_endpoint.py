import copy
import io
import logging
import socket
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as ksa_exc
import OpenSSL
from oslo_utils import importutils
from oslo_utils import netutils
import requests
import urllib.parse
from oslo_utils import encodeutils
from glanceclient.common import utils
from glanceclient import exc
@staticmethod
def parse_endpoint(endpoint):
    return netutils.urlsplit(endpoint)