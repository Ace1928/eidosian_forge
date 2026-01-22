import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
@utils.arg('name', metavar='<name>', help=_('Name of aggregate.'))
@utils.arg('availability_zone', metavar='<availability-zone>', default=None, nargs='?', help=_('The availability zone of the aggregate (optional).'))
def do_aggregate_create(cs, args):
    """Create a new aggregate with the specified details."""
    aggregate = cs.aggregates.create(args.name, args.availability_zone)
    _print_aggregate_details(cs, aggregate)