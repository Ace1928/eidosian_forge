from collections import OrderedDict
from typing import Dict, List, Tuple
import torch
from lightning_utilities.core.imports import RequirementCache
from torch.nn import Parameter
from typing_extensions import override
from pytorch_lightning.utilities.model_summary.model_summary import (
@property
def parameters_per_layer(self) -> List[int]:
    return [layer.average_shard_parameters for layer in self._layer_summary.values()]