import logging
import os
from argparse import Namespace
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Union
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import Module
from typing_extensions import override
from lightning_fabric.loggers.logger import Logger, rank_zero_experiment
from lightning_fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning_fabric.utilities.logger import _add_prefix, _convert_params, _flatten_dict
from lightning_fabric.utilities.logger import _sanitize_params as _utils_sanitize_params
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning_fabric.utilities.types import _PATH
from lightning_fabric.wrappers import _unwrap_objects
@property
def sub_dir(self) -> Optional[str]:
    """Gets the sub directory where the TensorBoard experiments are saved.

        Returns:
            The local path to the sub directory where the TensorBoard experiments are saved.

        """
    return self._sub_dir