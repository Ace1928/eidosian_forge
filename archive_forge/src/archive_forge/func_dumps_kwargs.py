import copy
import json
import warnings
from typing import Optional, Tuple
from scrapy.http.request import Request
from scrapy.utils.deprecate import create_deprecated_class
@property
def dumps_kwargs(self) -> dict:
    return self._dumps_kwargs