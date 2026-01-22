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
def get_snapshot_instance_export_location(self, snapshot, export_location_uuid, microversion=None):
    """Returns an export location by snapshot instance and its UUID.

        :param snapshot: str -- Name or ID of a snapshot instance.
        :param export_location_uuid: str -- UUID of an export location.
        :param microversion: API microversion to be used for request.
        """
    snapshot_raw = self.manila('snapshot-instance-export-location-show %(snapshot)s %(el_uuid)s' % {'snapshot': snapshot, 'el_uuid': export_location_uuid}, microversion=microversion)
    snapshot = output_parser.details(snapshot_raw)
    return snapshot