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
def is_message_deleted(self, message, microversion=None):
    """Indicates whether message is deleted or not.

        :param message: str -- ID of message
        """
    try:
        self.get_message(message, microversion=microversion)
        return False
    except tempest_lib_exc.NotFound:
        return True