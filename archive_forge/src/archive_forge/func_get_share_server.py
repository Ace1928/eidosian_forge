import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
@not_found_wrapper
def get_share_server(self, share_server, microversion=None):
    """Returns share server by its Name or ID."""
    share_server_raw = self.manila('share-server-show %s' % share_server, microversion=microversion)
    share_server = output_parser.details(share_server_raw)
    return share_server