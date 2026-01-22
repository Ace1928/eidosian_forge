from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.utils.framework import try_import_tf, try_import_torch
def logp(self, actions):
    a1, a2 = (actions[:, 0], actions[:, 1])
    a1_vec = torch.unsqueeze(a1.float(), 1)
    a1_logits, a2_logits = self.model.action_module(self.inputs, a1_vec)
    return TorchCategorical(a1_logits).logp(a1) + TorchCategorical(a2_logits).logp(a2)