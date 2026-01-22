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
def list_route_advertised_from_bgp_speaker(self, speaker_id, **_params):
    """Fetches a list of all routes advertised by BGP speaker."""
    return self.get(self.bgp_speaker_path % speaker_id + '/get_advertised_routes', params=_params)