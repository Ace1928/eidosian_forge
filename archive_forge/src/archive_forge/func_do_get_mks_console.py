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
@api_versions.wraps('2.8')
@utils.arg('server', metavar='<server>', help=_('Name or ID of server.'))
def do_get_mks_console(cs, args):
    """Get an MKS console to a server."""
    server = _find_server(cs, args.server)
    data = server.get_mks_console()
    print_console(cs, data)