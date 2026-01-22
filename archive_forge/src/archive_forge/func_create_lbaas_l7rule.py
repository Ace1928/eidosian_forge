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
def create_lbaas_l7rule(self, l7policy, body=None):
    """Creates rule for a certain L7 policy."""
    return self.post(self.lbaas_l7rules_path % l7policy, body=body)