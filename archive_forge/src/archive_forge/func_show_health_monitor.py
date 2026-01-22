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
def show_health_monitor(self, health_monitor, **_params):
    """Fetches information of a certain load balancer health monitor."""
    return self.get(self.health_monitor_path % health_monitor, params=_params)