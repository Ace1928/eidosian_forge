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
def update_packet_rate_limit_rule(self, rule, policy, body=None):
    """Updates a packet rate limit rule."""
    return self.put(self.qos_packet_rate_limit_rule_path % (policy, rule), body=body)