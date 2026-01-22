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
@utils.arg('name', metavar='<name>', help='Name of new QoS specifications.')
@utils.arg('metadata', metavar='<key=value>', nargs='+', default=[], help='QoS specifications.')
def do_qos_create(cs, args):
    """Creates a qos specs."""
    keypair = None
    if args.metadata is not None:
        keypair = shell_utils.extract_metadata(args)
    qos_specs = cs.qos_specs.create(args.name, keypair)
    shell_utils.print_qos_specs(qos_specs)