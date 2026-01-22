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
@utils.arg('--hypervisor', metavar='<hypervisor>', default=None, help=_('Type of hypervisor.'))
def do_agent_list(cs, args):
    """DEPRECATED List all builds."""
    _emit_agent_deprecation_warning()
    result = cs.agents.list(args.hypervisor)
    columns = ['Agent_id', 'Hypervisor', 'OS', 'Architecture', 'Version', 'Md5hash', 'Url']
    utils.print_list(result, columns)