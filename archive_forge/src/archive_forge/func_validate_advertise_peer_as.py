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
@validate(name=ADVERTISE_PEER_AS)
def validate_advertise_peer_as(advertise_peer_as):
    if not isinstance(advertise_peer_as, bool):
        raise ConfigTypeError(desc='Invalid type for advertise-peer-as, expected bool got %s' % type(advertise_peer_as))
    return advertise_peer_as