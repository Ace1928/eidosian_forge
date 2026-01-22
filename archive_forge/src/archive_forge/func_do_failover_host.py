import argparse
import collections
import copy
import os
from oslo_utils import strutils
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3 import availability_zones
@utils.arg('host', metavar='<hostname>', help='Host name.')
@utils.arg('--backend_id', metavar='<backend-id>', help='ID of backend to failover to (Default=None)')
def do_failover_host(cs, args):
    """Failover a replicating cinder-volume host."""
    cs.services.failover_host(args.host, args.backend_id)