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
@validate(name=LABEL_RANGE)
def validate_label_range(label_range):
    min_label, max_label = label_range
    if not min_label or not max_label or (not isinstance(min_label, numbers.Integral)) or (not isinstance(max_label, numbers.Integral)) or (min_label < 17) or (min_label >= max_label):
        raise ConfigValueError(desc='Invalid label_range configuration value: (%s).' % label_range)
    return label_range