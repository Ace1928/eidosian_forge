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
@forbidden_wrapper
def list_snapshot_instance_export_locations(self, snapshot_instance, columns=None, microversion=None):
    """List snapshot instance export locations.

        :param snapshot_instance: str -- Name or ID of a snapshot instance.
        :param columns: str -- comma separated string of columns.
            Example, "--columns uuid,path".
        :param microversion: API microversion to be used for request.
        """
    cmd = 'snapshot-instance-export-location-list %s' % snapshot_instance
    if columns is not None:
        cmd += ' --columns ' + columns
    export_locations_raw = self.manila(cmd, microversion=microversion)
    export_locations = utils.listing(export_locations_raw)
    return export_locations