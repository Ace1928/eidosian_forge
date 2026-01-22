from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
def do_endpoints(cs, args):
    """Discover endpoints that get returned from the authenticate services."""
    catalog = cs.keystone_client.service_catalog.catalog
    for e in catalog.get('serviceCatalog', catalog.get('catalog')):
        cliutils.print_dict(e['endpoints'][0], e['name'])