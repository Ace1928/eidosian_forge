import functools
from typing import Optional
import numpy as np
from ray.rllib.core.models.base import Model
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.specs.checker import SpecCheckingError
from ray.rllib.core.models.specs.specs_base import Spec
from ray.rllib.core.models.specs.specs_base import TensorSpec
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.torch.primitives import TorchCNNTranspose, TorchMLP
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
An MLPHead that implements floating log stds for Gaussian distributions.