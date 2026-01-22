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
def do_encryption_type_list(cs, args):
    """Shows encryption type details for volume types. Admin only."""
    result = cs.volume_encryption_types.list()
    shell_utils.print_list(result, ['Volume Type ID', 'Provider', 'Cipher', 'Key Size', 'Control Location'])