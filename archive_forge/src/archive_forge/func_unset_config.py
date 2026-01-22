from .. import errors
from ..utils.utils import (
from .base import DictType
from .healthcheck import Healthcheck
def unset_config(self, key):
    """ Remove the ``key`` property from the ``config`` dict. """
    if key in self.config:
        del self.config[key]