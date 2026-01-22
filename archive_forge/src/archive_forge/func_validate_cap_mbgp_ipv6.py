import abc
from abc import ABCMeta
from abc import abstractmethod
import functools
import numbers
import logging
import uuid
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import get_validator
from os_ken.services.protocols.bgp.base import RUNTIME_CONF_ERROR_CODE
from os_ken.services.protocols.bgp.base import validate
from os_ken.services.protocols.bgp.utils import validation
from os_ken.services.protocols.bgp.utils.validation import is_valid_asn
@validate(name=CAP_MBGP_IPV6)
def validate_cap_mbgp_ipv6(cmv6):
    if not isinstance(cmv6, bool):
        raise ConfigTypeError(desc='Invalid MP-BGP IPv6 capability settings: %s. Boolean value expected' % cmv6)
    return cmv6