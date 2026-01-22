import argparse
import errno
import glob
import logging
import logging.handlers
import os
import platform
import re
import shutil
import time
import traceback
from typing import Callable, List, Optional, Set
from ray._raylet import GcsClient
import ray._private.ray_constants as ray_constants
import ray._private.services as services
import ray._private.utils
from ray._private.ray_logging import setup_component_logger
def get_is_autoscaler_v2(self, gcs_address: Optional[str]) -> bool:
    """Check if autoscaler v2 is enabled."""
    if gcs_address is None:
        return False
    if not ray.experimental.internal_kv._internal_kv_initialized():
        gcs_client = GcsClient(address=gcs_address)
        ray.experimental.internal_kv._initialize_internal_kv(gcs_client)
    from ray.autoscaler.v2.utils import is_autoscaler_v2
    return is_autoscaler_v2()