import logging
import numbers
from os_ken.lib import ip
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv4
from os_ken.services.protocols.bgp.utils.validation import is_valid_asn
from os_ken.services.protocols.bgp import rtconf
from os_ken.services.protocols.bgp.rtconf.base import BaseConf
from os_ken.services.protocols.bgp.rtconf.base import BaseConfListener
from os_ken.services.protocols.bgp.rtconf.base import compute_optional_conf
from os_ken.services.protocols.bgp.rtconf.base import ConfigTypeError
from os_ken.services.protocols.bgp.rtconf.base import ConfigValueError
from os_ken.services.protocols.bgp.rtconf.base import MissingRequiredConf
from os_ken.services.protocols.bgp.rtconf.base import validate
@validate(name=REFRESH_STALEPATH_TIME)
def validate_refresh_stalepath_time(rst):
    if not isinstance(rst, numbers.Integral):
        raise ConfigTypeError(desc='Configuration value for %s has to be integral type' % REFRESH_STALEPATH_TIME)
    if rst < 0:
        raise ConfigValueError(desc='Invalid refresh stalepath time %s' % rst)
    return rst