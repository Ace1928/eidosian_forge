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
def wait_for_message_deletion(self, message, microversion=None):
    """Wait for message deletion by its ID.

        :param message: text -- ID of message
        """
    self.wait_for_resource_deletion(MESSAGE, res_id=message, interval=3, timeout=60, microversion=microversion)