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
def create_bgpvpn_port_assoc(self, bgpvpn, body=None):
    """Creates a new BGP VPN port association"""
    return self.post(self.bgpvpn_port_associations_path % bgpvpn, body=body)