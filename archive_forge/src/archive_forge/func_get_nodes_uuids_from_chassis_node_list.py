import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def get_nodes_uuids_from_chassis_node_list(self, chassis_uuid):
    chassis_node_list = self.list_node_chassis(chassis_uuid)
    return [x['UUID'] for x in chassis_node_list]