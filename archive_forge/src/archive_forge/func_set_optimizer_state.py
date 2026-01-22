import logging
import pathlib
from typing import (
from ray.rllib.core.learner.learner import (
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModule
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchDDPRLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import (
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.torch_utils import (
from ray.rllib.utils.typing import Optimizer, Param, ParamDict, TensorType
@override(Learner)
def set_optimizer_state(self, state: Mapping[str, Any]) -> None:
    for name, state_dict in state.items():
        if name not in self._named_optimizers:
            raise ValueError(f'Optimizer {name} in `state` is not known.Known optimizers are {self._named_optimizers.keys()}')
        optim = self._named_optimizers[name]
        state_dict_correct_device = copy_torch_tensors(state_dict, device=self._device)
        optim.load_state_dict(state_dict_correct_device)