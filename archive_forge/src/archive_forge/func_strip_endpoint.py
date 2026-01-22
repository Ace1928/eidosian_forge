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
def strip_endpoint(self, location):
    if location is None:
        message = _('Location not returned with 302')
        raise exc.InvalidEndpoint(message=message)
    if self.endpoint_override is not None and location.lower().startswith(self.endpoint_override.lower()):
        return location[len(self.endpoint_override):]
    else:
        return location