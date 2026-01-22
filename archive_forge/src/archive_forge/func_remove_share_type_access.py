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
def remove_share_type_access(self, share_type_name_or_id, project_id, microversion=None):
    data = dict(st=share_type_name_or_id, project=project_id)
    self.manila('type-access-remove %(st)s %(project)s' % data, microversion=microversion)