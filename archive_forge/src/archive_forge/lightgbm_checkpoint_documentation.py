import os
import tempfile
from typing import TYPE_CHECKING, Optional
import lightgbm
from ray.train._internal.framework_checkpoint import FrameworkCheckpoint
from ray.util.annotations import PublicAPI
Retrieve the LightGBM model stored in this checkpoint.