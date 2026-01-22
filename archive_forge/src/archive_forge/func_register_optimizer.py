import abc
import json
import logging
import pathlib
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, field
from typing import (
import ray
from ray.rllib.core.learner.reduce_result_dict_fn import _reduce_mean_results
from ray.rllib.core.learner.scaling_config import LearnerGroupScalingConfig
from ray.rllib.core.rl_module.marl_module import (
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, MultiAgentBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.debug import update_global_seed_if_necessary
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.minibatch_utils import (
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.serialization import serialize_type
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
def register_optimizer(self, *, module_id: ModuleID=ALL_MODULES, optimizer_name: str=DEFAULT_OPTIMIZER, optimizer: Optimizer, params: Sequence[Param], lr_or_lr_schedule: Optional[LearningRateOrSchedule]=None) -> None:
    """Registers an optimizer with a ModuleID, name, param list and lr-scheduler.

        Use this method in your custom implementations of either
        `self.configure_optimizers()` or `self.configure_optimzers_for_module()` (you
        should only override one of these!). If you register a learning rate Scheduler
        setting together with an optimizer, RLlib will automatically keep this
        optimizer's learning rate updated throughout the training process.
        Alternatively, you can construct your optimizers directly with a learning rate
        and manage learning rate scheduling or updating yourself.

        Args:
            module_id: The `module_id` under which to register the optimizer. If not
                provided, will assume ALL_MODULES.
            optimizer_name: The name (str) of the optimizer. If not provided, will
                assume DEFAULT_OPTIMIZER.
            optimizer: The already instantiated optimizer object to register.
            params: A list of parameters (framework-specific variables) that will be
                trained/updated
            lr_or_lr_schedule: An optional fixed learning rate or learning rate schedule
                setup. If provided, RLlib will automatically keep the optimizer's
                learning rate updated.
        """
    self._check_registered_optimizer(optimizer, params)
    full_registration_name = module_id + '_' + optimizer_name
    self._module_optimizers[module_id].append(full_registration_name)
    self._named_optimizers[full_registration_name] = optimizer
    self._optimizer_parameters[optimizer] = []
    for param in params:
        param_ref = self.get_param_ref(param)
        self._optimizer_parameters[optimizer].append(param_ref)
        self._params[param_ref] = param
    if lr_or_lr_schedule is not None:
        Scheduler.validate(fixed_value_or_schedule=lr_or_lr_schedule, setting_name='lr_or_lr_schedule', description='learning rate or schedule')
        scheduler = Scheduler(fixed_value_or_schedule=lr_or_lr_schedule, framework=self.framework, device=self._device)
        self._optimizer_lr_schedules[optimizer] = scheduler
        self._set_optimizer_lr(optimizer=optimizer, lr=scheduler.get_current_value())