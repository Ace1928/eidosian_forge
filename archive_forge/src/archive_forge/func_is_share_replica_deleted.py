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
def is_share_replica_deleted(self, replica, microversion=None):
    """Indicates whether a share replica is deleted or not.

        :param replica: str -- ID of share replica
        """
    try:
        self.get_share_replica(replica, microversion=microversion)
        return False
    except tempest_lib_exc.NotFound:
        return True