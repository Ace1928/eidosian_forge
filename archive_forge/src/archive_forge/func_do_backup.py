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
@utils.arg('name', metavar='<name>', help=_('Name of the backup image.'))
@utils.arg('backup_type', metavar='<backup-type>', help=_('The backup type, like "daily" or "weekly".'))
@utils.arg('rotation', metavar='<rotation>', help=_('Int parameter representing how many backups to keep around.'))
def do_backup(cs, args):
    """Backup a server by creating a 'backup' type snapshot."""
    result = _find_server(cs, args.server).backup(args.name, args.backup_type, args.rotation)
    if cs.api_version >= api_versions.APIVersion('2.45'):
        _print_image(_find_image(cs, result['image_id']))