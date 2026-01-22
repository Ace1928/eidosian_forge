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
@api_versions.wraps('2.10')
@utils.arg('keypair', metavar='<keypair>', help=_('Name of keypair.'))
@utils.arg('--user', metavar='<user-id>', default=None, help=_('ID of key-pair owner (Admin only).'))
def do_keypair_show(cs, args):
    """Show details about the given keypair."""
    keypair = cs.keypairs.get(args.keypair, args.user)
    _print_keypair(keypair)