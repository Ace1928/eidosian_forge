import argparse
import collections
import os
from oslo_utils import strutils
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3.shell_base import *  # noqa
from cinderclient.v3.shell_base import CheckSizeArgForCreate
@api_versions.wraps('3.3')
@utils.arg('message', metavar='<message>', help='ID of message.')
def do_message_show(cs, args):
    """Shows message details."""
    info = dict()
    message = shell_utils.find_message(cs, args.message)
    info.update(message._info)
    info.pop('links', None)
    shell_utils.print_dict(info)