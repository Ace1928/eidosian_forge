import copy
import hashlib
import logging
import os
import socket
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import importutils
import requests
from urllib import parse
from heatclient._i18n import _
from heatclient.common import utils
from heatclient import exc
def credentials_headers(self):
    return {}