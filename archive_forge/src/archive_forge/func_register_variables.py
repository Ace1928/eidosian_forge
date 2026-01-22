import contextlib
import gymnasium as gym
import re
from typing import Dict, List, Union
from ray.util import log_once
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict, TensorType
def register_variables(self, variables: List[TensorType]) -> None:
    """Register the given list of variables with this model."""
    if log_once('deprecated_tfmodelv2_register_variables'):
        deprecation_warning(old='TFModelV2.register_variables', error=False)
    self.var_list.extend(variables)