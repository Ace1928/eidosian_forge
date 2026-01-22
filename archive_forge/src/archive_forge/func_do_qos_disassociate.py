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
@utils.arg('qos_specs', metavar='<qos_specs>', help='ID of QoS specifications.')
@utils.arg('vol_type_id', metavar='<volume_type_id>', help='ID of volume type with which to associate QoS specifications.')
def do_qos_disassociate(cs, args):
    """Disassociates qos specs from specified volume type."""
    cs.qos_specs.disassociate(args.qos_specs, args.vol_type_id)