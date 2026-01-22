import os
import tempfile
from typing import TYPE_CHECKING, Optional
import xgboost
from ray.train._internal.framework_checkpoint import FrameworkCheckpoint
from ray.util.annotations import PublicAPI
Retrieve the XGBoost model stored in this checkpoint.