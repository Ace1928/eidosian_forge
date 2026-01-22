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
def list_network_loggable_resources(self, retrieve_all=True, **_params):
    """Fetch a list of supported resource types for network log."""
    return self.list('loggable_resources', self.network_loggables_path, retrieve_all, **_params)