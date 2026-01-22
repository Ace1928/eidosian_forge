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
@utils.arg('volume_type', metavar='<volume_type>', type=str, help='Name or ID of volume type.')
@utils.arg('provider', metavar='<provider>', type=str, help='The encryption provider format. For example, "luks" or "plain".')
@utils.arg('--cipher', metavar='<cipher>', type=str, required=False, default=None, help='The encryption algorithm or mode. For example, aes-xts-plain64. Default=None.')
@utils.arg('--key-size', metavar='<key_size>', type=int, required=False, default=None, help='Size of encryption key, in bits. For example, 128 or 256. Default=None.')
@utils.arg('--key_size', type=int, required=False, default=None, help=argparse.SUPPRESS)
@utils.arg('--control-location', metavar='<control_location>', choices=['front-end', 'back-end'], type=str, required=False, default='front-end', help='Notional service where encryption is performed. Valid values are "front-end" or "back-end". For example, front-end=Nova. Default is "front-end".')
@utils.arg('--control_location', type=str, required=False, default='front-end', help=argparse.SUPPRESS)
def do_encryption_type_create(cs, args):
    """Creates encryption type for a volume type. Admin only."""
    volume_type = shell_utils.find_volume_type(cs, args.volume_type)
    body = {'provider': args.provider, 'cipher': args.cipher, 'key_size': args.key_size, 'control_location': args.control_location}
    result = cs.volume_encryption_types.create(volume_type, body)
    shell_utils.print_volume_encryption_type_list([result])