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
@utils.arg('volume', metavar='<volume>', help='Name or ID of the volume to unmanage.')
def do_unmanage(cs, args):
    """Stop managing a volume."""
    volume = utils.find_volume(cs, args.volume)
    cs.volumes.unmanage(volume.id)