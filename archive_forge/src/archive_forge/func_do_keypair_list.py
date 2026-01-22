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
@api_versions.wraps('2.35')
@utils.arg('--user', metavar='<user-id>', default=None, help=_('List key-pairs of specified user ID (Admin only).'))
@utils.arg('--marker', dest='marker', metavar='<marker>', default=None, help=_('The last keypair of the previous page; displays list of keypairs after "marker".'))
@utils.arg('--limit', dest='limit', metavar='<limit>', type=int, default=None, help=_("Maximum number of keypairs to display. If limit is bigger than 'CONF.api.max_limit' option of Nova API, limit 'CONF.api.max_limit' will be used instead."))
def do_keypair_list(cs, args):
    """Print a list of keypairs for a user"""
    keypairs = cs.keypairs.list(args.user, args.marker, args.limit)
    columns = _get_keypairs_list_columns(cs, args)
    utils.print_list(keypairs, columns)