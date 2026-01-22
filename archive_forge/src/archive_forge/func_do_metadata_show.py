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
@utils.arg('volume', metavar='<volume>', help='ID of volume.')
def do_metadata_show(cs, args):
    """Shows volume metadata."""
    volume = utils.find_volume(cs, args.volume)
    shell_utils.print_dict(volume._info['metadata'], 'Metadata-property')