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
@utils.arg('host', metavar='<host>', help='The hypervisor hostname (or pattern) to search for. WARNING: Use a fully qualified domain name if you only want to live migrate from a specific host.')
@utils.arg('--target-host', metavar='<target_host>', default=None, help=_('Name of target host. If no host is specified, the scheduler will choose one.'))
@utils.arg('--block-migrate', action='store_true', default=False, help=_('Enable block migration. (Default=False)'), start_version='2.0', end_version='2.24')
@utils.arg('--block-migrate', action='store_true', default='auto', help=_('Enable block migration. (Default=auto)'), start_version='2.25')
@utils.arg('--disk-over-commit', action='store_true', default=False, help=_('Enable disk overcommit.'), start_version='2.0', end_version='2.24')
@utils.arg('--max-servers', type=int, dest='max_servers', metavar='<max_servers>', help='Maximum number of servers to live migrate simultaneously')
@utils.arg('--force', dest='force', action='store_true', default=False, help=_('Force a live-migration by not verifying the provided destination host by the scheduler. WARNING: This could result in failures to actually live migrate the servers to the specified host. It is recommended to either not specify a host so that the scheduler will pick one, or specify a host without --force.'), start_version='2.30', end_version='2.67')
@utils.arg('--strict', dest='strict', action='store_true', default=False, help=_('live Evacuate host with exact hypervisor hostname match'))
def do_host_evacuate_live(cs, args):
    """Live migrate all instances off the specified host
    to other available hosts.
    """
    response = []
    migrating = 0
    for server in _hyper_servers(cs, args.host, args.strict):
        response.append(_server_live_migrate(cs, server, args))
        migrating = migrating + 1
        if args.max_servers is not None and migrating >= args.max_servers:
            break
    utils.print_list(response, ['Server UUID', 'Live Migration Accepted', 'Error Message'])