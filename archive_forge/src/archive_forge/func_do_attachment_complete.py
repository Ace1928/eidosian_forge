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
@api_versions.wraps('3.44')
@utils.arg('attachment', metavar='<attachment>', nargs='+', help='ID of attachment or attachments to delete.')
def do_attachment_complete(cs, args):
    """Complete an attachment for a cinder volume."""
    for attachment in args.attachment:
        cs.attachments.complete(attachment)