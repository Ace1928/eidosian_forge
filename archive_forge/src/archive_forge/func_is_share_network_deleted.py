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
def is_share_network_deleted(self, share_network, microversion=None):
    """Says whether share network is deleted or not.

        :param share_network: text -- Name or ID of share network
        """
    share_types = self.list_share_networks(True, microversion=microversion)
    for list_element in share_types:
        if share_network in (list_element['id'], list_element['name']):
            return False
    return True