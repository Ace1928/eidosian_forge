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
def show_sfc_port_pair_group(self, port_pair_group, **_params):
    """Fetches information of a certain Port Pair Group."""
    return self.get(self.sfc_port_pair_group_path % port_pair_group, params=_params)