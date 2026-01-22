import ncclient
import ncclient.manager
import ncclient.xml_
from os_ken import exception as os_ken_exc
from os_ken.lib import of_config
from os_ken.lib.of_config import constants as ofc_consts
from os_ken.lib.of_config import classes as ofc
def get_ns_tag(tag):
    if tag[0] == '{':
        return tuple(tag[1:].split('}', 1))
    return (None, tag)