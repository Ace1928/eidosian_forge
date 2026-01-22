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
@utils.arg('host', metavar='<host>', help='The hypervisor hostname (or pattern) to search for. WARNING: Use a fully qualified domain name if you only want to update metadata for servers on a specific host.')
@utils.arg('action', metavar='<action>', choices=['set', 'delete'], help=_("Actions: 'set' or 'delete'"))
@utils.arg('metadata', metavar='<key=value>', nargs='+', action='append', default=[], help=_('Metadata to set or delete (only key is necessary on delete)'))
@utils.arg('--strict', dest='strict', action='store_true', default=False, help=_('Set host-meta to the hypervisor with exact hostname match'))
def do_host_meta(cs, args):
    """Set or Delete metadata on all instances of a host."""
    for server in _hyper_servers(cs, args.host, args.strict):
        metadata = _extract_metadata(args)
        if args.action == 'set':
            cs.servers.set_meta(server['uuid'], metadata)
        elif args.action == 'delete':
            cs.servers.delete_meta(server['uuid'], metadata.keys())