import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from functools import partial
import logging
from typing import Dict, List, Optional, Tuple, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.dqn.dqn_tf_policy import (
from ray.rllib.algorithms.sac.sac_tf_model import SACTFModel
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import (
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import get_variable, try_import_tf
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.tf_utils import huber_loss, make_tf_callable
from ray.rllib.utils.typing import (
def compute_and_clip_gradients(policy: Policy, optimizer: LocalOptimizer, loss: TensorType) -> ModelGradients:
    """Gradients computing function (from loss tensor, using local optimizer).

    Note: For SAC, optimizer and loss are ignored b/c we have 3
    losses and 3 local optimizers (all stored in policy).
    `optimizer` will be used, though, in the tf-eager case b/c it is then a
    fake optimizer (OptimizerWrapper) object with a `tape` property to
    generate a GradientTape object for gradient recording.

    Args:
        policy: The Policy object that generated the loss tensor and
            that holds the given local optimizer.
        optimizer: The tf (local) optimizer object to
            calculate the gradients with.
        loss: The loss tensor for which gradients should be
            calculated.

    Returns:
        ModelGradients: List of the possibly clipped gradients- and variable
            tuples.
    """
    if policy.config['framework'] == 'tf2':
        tape = optimizer.tape
        pol_weights = policy.model.policy_variables()
        actor_grads_and_vars = list(zip(tape.gradient(policy.actor_loss, pol_weights), pol_weights))
        q_weights = policy.model.q_variables()
        if policy.config['twin_q']:
            half_cutoff = len(q_weights) // 2
            grads_1 = tape.gradient(policy.critic_loss[0], q_weights[:half_cutoff])
            grads_2 = tape.gradient(policy.critic_loss[1], q_weights[half_cutoff:])
            critic_grads_and_vars = list(zip(grads_1, q_weights[:half_cutoff])) + list(zip(grads_2, q_weights[half_cutoff:]))
        else:
            critic_grads_and_vars = list(zip(tape.gradient(policy.critic_loss[0], q_weights), q_weights))
        alpha_vars = [policy.model.log_alpha]
        alpha_grads_and_vars = list(zip(tape.gradient(policy.alpha_loss, alpha_vars), alpha_vars))
    else:
        actor_grads_and_vars = policy._actor_optimizer.compute_gradients(policy.actor_loss, var_list=policy.model.policy_variables())
        q_weights = policy.model.q_variables()
        if policy.config['twin_q']:
            half_cutoff = len(q_weights) // 2
            base_q_optimizer, twin_q_optimizer = policy._critic_optimizer
            critic_grads_and_vars = base_q_optimizer.compute_gradients(policy.critic_loss[0], var_list=q_weights[:half_cutoff]) + twin_q_optimizer.compute_gradients(policy.critic_loss[1], var_list=q_weights[half_cutoff:])
        else:
            critic_grads_and_vars = policy._critic_optimizer[0].compute_gradients(policy.critic_loss[0], var_list=q_weights)
        alpha_grads_and_vars = policy._alpha_optimizer.compute_gradients(policy.alpha_loss, var_list=[policy.model.log_alpha])
    if policy.config['grad_clip']:
        clip_func = partial(tf.clip_by_norm, clip_norm=policy.config['grad_clip'])
    else:
        clip_func = tf.identity
    policy._actor_grads_and_vars = [(clip_func(g), v) for g, v in actor_grads_and_vars if g is not None]
    policy._critic_grads_and_vars = [(clip_func(g), v) for g, v in critic_grads_and_vars if g is not None]
    policy._alpha_grads_and_vars = [(clip_func(g), v) for g, v in alpha_grads_and_vars if g is not None]
    grads_and_vars = policy._actor_grads_and_vars + policy._critic_grads_and_vars + policy._alpha_grads_and_vars
    return grads_and_vars