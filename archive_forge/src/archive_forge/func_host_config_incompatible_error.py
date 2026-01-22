from .. import errors
from ..utils.utils import (
from .base import DictType
from .healthcheck import Healthcheck
def host_config_incompatible_error(param, param_value, incompatible_param):
    error_msg = '"{1}" {0} is incompatible with {2}'
    return errors.InvalidArgument(error_msg.format(param, param_value, incompatible_param))