from .. import errors
from ..utils.utils import (
from .base import DictType
from .healthcheck import Healthcheck
def host_config_type_error(param, param_value, expected):
    error_msg = 'Invalid type for {0} param: expected {1} but found {2}'
    return TypeError(error_msg.format(param, expected, type(param_value)))