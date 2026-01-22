import contextlib
import io
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import get_full_repo_name
from packaging import version
from .debug_utils import DebugOption
from .trainer_utils import (
from .utils import (
from .utils.generic import strtobool
from .utils.import_utils import is_optimum_neuron_available
@property
def should_save(self):
    """
        Whether or not the current process should write to disk, e.g., to save models and checkpoints.
        """
    if self.save_on_each_node:
        return self.local_process_index == 0
    elif is_sagemaker_mp_enabled():
        return smp.rank() == 0
    else:
        return self.process_index == 0