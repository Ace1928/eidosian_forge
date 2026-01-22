import copy
import json
import logging
import os
from typing import Optional, Tuple
import ray
from ray.serve._private.common import ServeComponentType
from ray.serve._private.constants import (
from ray.serve.schema import EncodingType, LoggingConfig
Format the log record into the format string.

        Args:
            record: The log record to be formatted.

            Returns:
                The formatted log record in string format.
        