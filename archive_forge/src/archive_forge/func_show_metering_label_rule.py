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
def show_metering_label_rule(self, metering_label_rule, **_params):
    """Fetches information of a certain metering label rule."""
    return self.get(self.metering_label_rule_path % metering_label_rule, params=_params)