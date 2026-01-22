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
def do_server_migration_list(cs, args):
    """Get the migrations list of specified server."""
    server = _find_server(cs, args.server)
    migrations = cs.server_migrations.list(server)
    fields = ['Id', 'Source Node', 'Dest Node', 'Source Compute', 'Dest Compute', 'Dest Host', 'Status', 'Server UUID', 'Created At', 'Updated At']
    format_name = ['Total Memory Bytes', 'Processed Memory Bytes', 'Remaining Memory Bytes', 'Total Disk Bytes', 'Processed Disk Bytes', 'Remaining Disk Bytes']
    format_key = ['memory_total_bytes', 'memory_processed_bytes', 'memory_remaining_bytes', 'disk_total_bytes', 'disk_processed_bytes', 'disk_remaining_bytes']
    if cs.api_version >= api_versions.APIVersion('2.80'):
        fields.append('Project ID')
        fields.append('User ID')
    formatters = map(lambda field: utils.make_field_formatter(field)[1], format_key)
    formatters = dict(zip(format_name, formatters))
    utils.print_list(migrations, fields + format_name, formatters)