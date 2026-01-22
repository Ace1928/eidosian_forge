import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def get_chassis_uuids_from_chassis_list(self):
    chassis_list = self.list_chassis()
    return [x['UUID'] for x in chassis_list]