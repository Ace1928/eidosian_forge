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
@utils.arg('qos_specs', metavar='<qos_specs>', help='ID of QoS specifications on which to operate.')
def do_qos_disassociate_all(cs, args):
    """Disassociates qos specs from all its associations."""
    cs.qos_specs.disassociate_all(args.qos_specs)