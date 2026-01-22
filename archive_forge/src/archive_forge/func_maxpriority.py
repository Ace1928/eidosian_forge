from __future__ import annotations
import copy
import json
from importlib import import_module
from pprint import pformat
from types import ModuleType
from typing import (
from scrapy.settings import default_settings
def maxpriority(self) -> int:
    """
        Return the numerical value of the highest priority present throughout
        all settings, or the numerical value for ``default`` from
        :attr:`~scrapy.settings.SETTINGS_PRIORITIES` if there are no settings
        stored.
        """
    if len(self) > 0:
        return max((cast(int, self.getpriority(name)) for name in self))
    return get_settings_priority('default')