import json
import logging
import pathlib
from typing import (
from ray.rllib.core.learner.learner import (
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.core.rl_module.tf.tf_rl_module import TfRLModule
from ray.rllib.policy.eager_tf_policy import _convert_to_tf
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.serialization import convert_numpy_to_python_primitives
from ray.rllib.utils.typing import Optimizer, Param, ParamDict, TensorType
def helper(_batch):
    _batch = NestedDict(_batch)
    with tf.GradientTape(persistent=True) as tape:
        fwd_out = self._module.forward_train(_batch)
        loss_per_module = self.compute_loss(fwd_out=fwd_out, batch=_batch)
    gradients = self.compute_gradients(loss_per_module, gradient_tape=tape)
    del tape
    postprocessed_gradients = self.postprocess_gradients(gradients)
    self.apply_gradients(postprocessed_gradients)
    return (fwd_out, loss_per_module, dict(self._metrics))