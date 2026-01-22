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
def get_lbaas_agent_hosting_loadbalancer(self, loadbalancer, **_params):
    """Fetches a loadbalancer agent hosting a loadbalancer."""
    return self.get((self.lbaas_loadbalancer_path + self.LOADBALANCER_HOSTING_AGENT) % loadbalancer, params=_params)