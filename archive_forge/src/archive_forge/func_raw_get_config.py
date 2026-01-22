import ncclient
import ncclient.manager
import ncclient.xml_
from os_ken import exception as os_ken_exc
from os_ken.lib import of_config
from os_ken.lib.of_config import constants as ofc_consts
from os_ken.lib.of_config import classes as ofc
def raw_get_config(self, source, filter=None):
    reply = self.netconf.get_config(source, filter)
    return self._find_capable_switch_xml(reply.data_ele)