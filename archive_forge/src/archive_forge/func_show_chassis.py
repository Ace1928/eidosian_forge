import configparser
import os
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import ironicclient.tests.functional.utils as utils
def show_chassis(self, chassis_id, params=''):
    chassis_show = self.ironic('chassis-show', params='{0} {1}'.format(chassis_id, params))
    return utils.get_dict_from_output(chassis_show)