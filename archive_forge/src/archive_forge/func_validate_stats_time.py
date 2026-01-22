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
@validate(name=ConfWithStats.STATS_TIME)
def validate_stats_time(stats_time):
    if not isinstance(stats_time, numbers.Integral):
        raise ConfigTypeError(desc='Statistics log timer value has to be of integral type but got: %r' % stats_time)
    if stats_time < 10:
        raise ConfigValueError(desc='Statistics log timer cannot be set to less then 10 sec, given timer value %s.' % stats_time)
    return stats_time