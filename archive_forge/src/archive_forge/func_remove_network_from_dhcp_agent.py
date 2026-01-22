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
def remove_network_from_dhcp_agent(self, dhcp_agent, network_id):
    """Remove a network from dhcp agent."""
    return self.delete((self.agent_path + self.DHCP_NETS + '/%s') % (dhcp_agent, network_id))