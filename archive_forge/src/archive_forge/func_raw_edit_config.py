import ncclient
import ncclient.manager
import ncclient.xml_
from os_ken import exception as os_ken_exc
from os_ken.lib import of_config
from os_ken.lib.of_config import constants as ofc_consts
from os_ken.lib.of_config import classes as ofc
def raw_edit_config(self, target, config, default_operation=None, test_option=None, error_option=None):
    self.netconf.edit_config(target, config, default_operation, test_option, error_option)