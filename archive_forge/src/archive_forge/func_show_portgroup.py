import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def show_portgroup(self, portgroup_id, params=''):
    """Show detailed information about a port group."""
    portgroup_show = self.ironic('portgroup-show', flags=self.pg_api_ver, params='{0} {1}'.format(portgroup_id, params))
    return utils.get_dict_from_output(portgroup_show)