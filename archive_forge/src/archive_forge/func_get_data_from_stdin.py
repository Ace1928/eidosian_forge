import argparse
from base64 import b64encode
import logging
import os
import sys
from cinderclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack.image import image_signer
from osc_lib.api import utils as api_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.common import progressbar
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
def get_data_from_stdin():
    try:
        os.fstat(0)
    except OSError:
        return None
    if not sys.stdin.isatty():
        image = sys.stdin
        if hasattr(sys.stdin, 'buffer'):
            image = sys.stdin.buffer
        if msvcrt:
            msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        return image
    else:
        return None