import inspect
import itertools
import logging
import re
import time
import urllib.parse as urlparse
import debtcollector.renames
from keystoneauth1 import exceptions as ksa_exc
import requests
from neutronclient._i18n import _
from neutronclient import client
from neutronclient.common import exceptions
from neutronclient.common import extension as client_extension
from neutronclient.common import serializer
from neutronclient.common import utils
def connect_network_gateway(self, gateway_id, body=None):
    """Connect a network gateway to the specified network."""
    base_uri = self.network_gateway_path % gateway_id
    return self.put('%s/connect_network' % base_uri, body=body)