from __future__ import annotations
import copy
import json
from importlib import import_module
from pprint import pformat
from types import ModuleType
from typing import (
from scrapy.settings import default_settings
def setdict(self, values: _SettingsInputT, priority: Union[int, str]='project') -> None:
    self.update(values, priority)