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
@validate(name=SITE_OF_ORIGINS)
def validate_soo_list(soo_list):
    if not isinstance(soo_list, list):
        raise ConfigTypeError(conf_name=SITE_OF_ORIGINS, conf_value=soo_list)
    if len(soo_list) > MAX_NUM_SOO:
        raise ConfigValueError(desc='Max. SOO is limited to %s' % MAX_NUM_SOO)
    if not all((validation.is_valid_ext_comm_attr(attr) for attr in soo_list)):
        raise ConfigValueError(conf_name=SITE_OF_ORIGINS, conf_value=soo_list)
    unique_rts = set(soo_list)
    if len(unique_rts) != len(soo_list):
        raise ConfigValueError(desc='Duplicate value provided in %s' % soo_list)
    return soo_list