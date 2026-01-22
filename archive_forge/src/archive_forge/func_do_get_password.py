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
@utils.arg('server', metavar='<server>', help=_('Name or ID of server.'))
@utils.arg('private_key', metavar='<private-key>', help=_('Private key (used locally to decrypt password) (Optional). When specified, the command displays the clear (decrypted) VM password. When not specified, the ciphered VM password is displayed.'), nargs='?', default=None)
def do_get_password(cs, args):
    """Get the admin password for a server. This operation calls the metadata
    service to query metadata information and does not read password
    information from the server itself.
    """
    server = _find_server(cs, args.server)
    data = server.get_password(args.private_key)
    print(data)