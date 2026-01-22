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
@validate(name=CAP_MBGP_L2VPNFS)
def validate_cap_mbgp_l2vpnfs(cml2fs):
    if not isinstance(cml2fs, bool):
        raise ConfigTypeError(desc='Invalid MP-BGP L2VPN Flow Specification capability settings: %s. Boolean value expected' % cml2fs)
    return cml2fs