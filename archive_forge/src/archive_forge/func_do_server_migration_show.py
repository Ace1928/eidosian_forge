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
@api_versions.wraps('2.23')
@utils.arg('server', metavar='<server>', help=_('Name or ID of server.'))
@utils.arg('migration', metavar='<migration>', help=_('ID of migration.'))
def do_server_migration_show(cs, args):
    """Get the migration of specified server."""
    server = _find_server(cs, args.server)
    migration = cs.server_migrations.get(server, args.migration)
    utils.print_dict(migration.to_dict())