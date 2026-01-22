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
@api_versions.wraps('3.0')
def do_api_version(cs, args):
    """Display the server API version information."""
    columns = ['ID', 'Status', 'Version', 'Min_version']
    response = cs.services.server_api_version()
    shell_utils.print_list(response, columns)