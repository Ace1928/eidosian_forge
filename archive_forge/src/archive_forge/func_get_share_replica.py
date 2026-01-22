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
def get_share_replica(self, replica, microversion=None):
    cmd = 'share-replica-show %s' % replica
    replica = self.manila(cmd, microversion=microversion)
    return output_parser.details(replica)