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
def get_share_export_location(self, share, export_location_uuid, microversion=None):
    """Returns an export location by share and its UUID.

        :param share: str -- Name or ID of a share.
        :param export_location_uuid: str -- UUID of an export location.
        :param microversion: API microversion to be used for request.
        """
    share_raw = self.manila('share-export-location-show %(share)s %(el_uuid)s' % {'share': share, 'el_uuid': export_location_uuid}, microversion=microversion)
    share = output_parser.details(share_raw)
    return share