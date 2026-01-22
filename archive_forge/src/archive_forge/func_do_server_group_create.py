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
@api_versions.wraps('2.64')
@utils.arg('name', metavar='<name>', help=_('Server group name.'))
@utils.arg('policy', metavar='<policy>', help=_('Policy for the server group.'))
@utils.arg('--rule', metavar='<key=value>', dest='rules', action='append', default=[], help=_('A rule for the policy. Currently, only the "max_server_per_host" rule is supported for the "anti-affinity" policy.'))
def do_server_group_create(cs, args):
    """Create a new server group with the specified details."""
    rules = _meta_parsing(args.rules)
    server_group = cs.server_groups.create(name=args.name, policy=args.policy, rules=rules)
    _print_server_group_details(cs, [server_group])