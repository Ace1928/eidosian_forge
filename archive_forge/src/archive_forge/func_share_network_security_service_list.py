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
def share_network_security_service_list(self, share_network_id, microversion=None):
    cmd = 'share-network-security-service-list %s' % share_network_id
    share_networks_raw = self.manila(cmd, microversion=microversion)
    network_services = utils.listing(share_networks_raw)
    return network_services