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
@validate(name=BGP_CONN_RETRY_TIME)
def validate_bgp_conn_retry_time(bgp_conn_retry_time):
    if not isinstance(bgp_conn_retry_time, numbers.Integral):
        raise ConfigTypeError(desc='Invalid bgp conn. retry time configuration value %s' % bgp_conn_retry_time)
    if bgp_conn_retry_time < 10:
        raise ConfigValueError(desc='Invalid bgp connection retry time configuration value %s' % bgp_conn_retry_time)
    return bgp_conn_retry_time