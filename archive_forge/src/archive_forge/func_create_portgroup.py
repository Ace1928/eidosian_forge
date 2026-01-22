import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def create_portgroup(self, node_id, params=''):
    """Create a new portgroup."""
    portgroup = self.ironic('portgroup-create', flags=self.pg_api_ver, params='--node {0} {1}'.format(node_id, params))
    if not portgroup:
        self.fail('Ironic portgroup failed to create!')
    portgroup = utils.get_dict_from_output(portgroup)
    self.addCleanup(self.delete_portgroup, portgroup['uuid'], ignore_exceptions=True)
    return portgroup