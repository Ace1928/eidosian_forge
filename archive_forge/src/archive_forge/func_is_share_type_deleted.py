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
def is_share_type_deleted(self, share_type, microversion=None):
    """Says whether share type is deleted or not.

        :param share_type: text -- Name or ID of share type
        """
    share_types = self.list_share_types(list_all=True, microversion=microversion)
    for list_element in share_types:
        if share_type in (list_element['ID'], list_element['Name']):
            return False
    return True