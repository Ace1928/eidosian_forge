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
def show_service_profile(self, flavor_profile, **_params):
    """Fetches information for a certain Neutron service flavor profile."""
    return self.get(self.service_profile_path % flavor_profile, params=_params)