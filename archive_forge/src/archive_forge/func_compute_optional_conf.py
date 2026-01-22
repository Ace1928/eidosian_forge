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
def compute_optional_conf(conf_name, default_value, **all_config):
    """Returns *conf_name* settings if provided in *all_config*, else returns
     *default_value*.

    Validates *conf_name* value if provided.
    """
    conf_value = all_config.get(conf_name)
    if conf_value is not None:
        conf_value = get_validator(conf_name)(conf_value)
    else:
        conf_value = default_value
    return conf_value