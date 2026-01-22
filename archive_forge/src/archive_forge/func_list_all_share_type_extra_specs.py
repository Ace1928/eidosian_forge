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
def list_all_share_type_extra_specs(self, microversion=None):
    """List extra specs for all share types."""
    extra_specs_raw = self.manila('extra-specs-list', microversion=microversion)
    extra_specs = utils.listing(extra_specs_raw)
    return extra_specs