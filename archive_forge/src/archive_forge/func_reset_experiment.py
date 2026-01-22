import logging
import os
from argparse import Namespace
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Union
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import Module
from typing_extensions import override
from lightning_fabric.utilities.logger import _add_prefix, _convert_params, _flatten_dict
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_only
def reset_experiment(self) -> None:
    self._experiment = None