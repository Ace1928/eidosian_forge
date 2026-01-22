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
@utils.arg('volume', metavar='<volume>', help='ID of volume to update.')
@utils.arg('read_only', metavar='<True|true|False|false>', choices=['True', 'true', 'False', 'false'], help='Enables or disables update of volume to read-only access mode.')
def do_readonly_mode_update(cs, args):
    """Updates volume read-only access-mode flag."""
    volume = utils.find_volume(cs, args.volume)
    cs.volumes.update_readonly_flag(volume, strutils.bool_from_string(args.read_only, strict=True))