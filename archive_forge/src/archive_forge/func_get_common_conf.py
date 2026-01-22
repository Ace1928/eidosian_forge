from os_ken.lib import hub
from os_ken.services.protocols.bgp.api.base import register
from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
from os_ken.services.protocols.bgp.rtconf.common import CommonConf
@register(name='comm_conf.get')
def get_common_conf():
    comm_conf = CORE_MANAGER.common_conf
    return comm_conf.settings