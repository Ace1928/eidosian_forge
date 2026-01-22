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
def should_update_filenames(self, last_file_updated_time: float) -> bool:
    """Return true if filenames should be updated.

        This method is used to apply the backpressure on file updates because
        that requires heavy glob operations which use lots of CPUs.

        Args:
            last_file_updated_time: The last time filenames are updated.

        Returns:
            True if filenames should be updated. False otherwise.
        """
    elapsed_seconds = float(time.time() - last_file_updated_time)
    return len(self.log_filenames) < RAY_LOG_MONITOR_MANY_FILES_THRESHOLD or elapsed_seconds > LOG_NAME_UPDATE_INTERVAL_S