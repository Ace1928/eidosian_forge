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
@utils.arg('hypervisor', metavar='<hypervisor>', help=_('Name or ID of the hypervisor. Starting with microversion 2.53 the ID must be a UUID.'))
def do_hypervisor_uptime(cs, args):
    """Display the uptime of the specified hypervisor."""
    hyper = _find_hypervisor(cs, args.hypervisor)
    hyper = cs.hypervisors.uptime(hyper)
    utils.print_dict(hyper.to_dict())