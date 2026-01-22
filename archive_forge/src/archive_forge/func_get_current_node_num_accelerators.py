import os
import re
import glob
import requests
import logging
from typing import Dict, Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
@staticmethod
def get_current_node_num_accelerators() -> int:
    """Attempt to detect the number of TPUs on this machine.

        TPU chips are represented as devices within `/dev/`, either as
        `/dev/accel*` or `/dev/vfio/*`.

        Returns:
            The number of TPUs if any were detected, otherwise 0.
        """
    accel_files = glob.glob('/dev/accel*')
    if accel_files:
        return len(accel_files)
    try:
        vfio_entries = os.listdir('/dev/vfio')
        numeric_entries = [int(entry) for entry in vfio_entries if entry.isdigit()]
        return len(numeric_entries)
    except FileNotFoundError as e:
        logger.debug('Failed to detect number of TPUs: %s', e)
        return 0