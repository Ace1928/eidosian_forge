import logging
import os
import time
from collections import namedtuple
from numbers import Number
from typing import Any, Dict, Optional
import ray
from ray._private.resource_spec import NODE_ID_PREFIX
def get_res_total(self, key):
    return self.custom_resources.get(key, 0) + self.extra_custom_resources.get(key, 0)