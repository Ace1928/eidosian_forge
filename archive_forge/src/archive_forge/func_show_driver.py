import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def show_driver(self, driver_name):
    driver_show = self.ironic('driver-show', params=driver_name)
    return utils.get_dict_from_output(driver_show)