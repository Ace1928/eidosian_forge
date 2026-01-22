import os
import tempfile
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional
import torch
from ray.air._internal.torch_utils import (
from ray.train._internal.framework_checkpoint import FrameworkCheckpoint
from ray.util.annotations import PublicAPI
Retrieve the model stored in this checkpoint.

        Args:
            model: If the checkpoint contains a model state dict, and not
                the model itself, then the state dict will be loaded to this
                ``model``. Otherwise, the model will be discarded.
        