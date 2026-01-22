import copy
import json
import logging
import os
from typing import Any, Dict
import yaml
from ray.autoscaler._private.loader import load_function_or_class
Retrieve a node provider.

    This is an INTERNAL API. It is not allowed to call this from any Ray
    package outside the autoscaler.
    