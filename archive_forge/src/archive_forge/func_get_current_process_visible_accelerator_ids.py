import os
import re
import glob
import requests
import logging
from typing import Dict, Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
@staticmethod
def get_current_process_visible_accelerator_ids() -> Optional[List[str]]:
    tpu_visible_chips = os.environ.get(TPUAcceleratorManager.get_visible_accelerator_ids_env_var(), None)
    if tpu_visible_chips is None:
        return None
    if tpu_visible_chips == '':
        return []
    return list(tpu_visible_chips.split(','))