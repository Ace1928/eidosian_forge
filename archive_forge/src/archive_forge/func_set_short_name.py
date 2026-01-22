import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def set_short_name(self, short_name):
    self._short_name = short_name