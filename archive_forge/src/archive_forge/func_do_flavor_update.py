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
@api_versions.wraps('2.55')
@utils.arg('flavor', metavar='<flavor>', help=_('Name or ID of the flavor to update.'))
@utils.arg('description', metavar='<description>', help=_('A free form description of the flavor. Limited to 65535 characters in length. Only printable characters are allowed.'))
def do_flavor_update(cs, args):
    """Update the description of an existing flavor."""
    flavorid = _find_flavor(cs, args.flavor)
    flavor = cs.flavors.update(flavorid, args.description)
    _print_flavor_list(cs, [flavor])