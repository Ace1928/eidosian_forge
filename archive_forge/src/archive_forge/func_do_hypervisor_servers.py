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
@utils.arg('hostname', metavar='<hostname>', help=_('The hypervisor hostname (or pattern) to search for.'))
def do_hypervisor_servers(cs, args):
    """List servers belonging to specific hypervisors."""
    hypers = cs.hypervisors.search(args.hostname, servers=True)

    class InstanceOnHyper(object):

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    instances = []
    for hyper in hypers:
        hyper_host = hyper.hypervisor_hostname
        hyper_id = hyper.id
        if hasattr(hyper, 'servers'):
            instances.extend([InstanceOnHyper(id=serv['uuid'], name=serv['name'], hypervisor_hostname=hyper_host, hypervisor_id=hyper_id) for serv in hyper.servers])
    utils.print_list(instances, ['ID', 'Name', 'Hypervisor ID', 'Hypervisor Hostname'])