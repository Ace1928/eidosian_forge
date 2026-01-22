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
@validate(name=CAP_MBGP_IPV4FS)
def validate_cap_mbgp_ipv4fs(cmv4fs):
    if not isinstance(cmv4fs, bool):
        raise ConfigTypeError(desc='Invalid MP-BGP IPv4 Flow Specification capability settings: %s. Boolean value expected' % cmv4fs)
    return cmv4fs