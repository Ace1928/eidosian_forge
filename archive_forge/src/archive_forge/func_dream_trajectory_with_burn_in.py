import re
import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.dreamerv3.tf.models.disagree_networks import DisagreeNetworks
from ray.rllib.algorithms.dreamerv3.tf.models.actor_network import ActorNetwork
from ray.rllib.algorithms.dreamerv3.tf.models.critic_network import CriticNetwork
from ray.rllib.algorithms.dreamerv3.tf.models.world_model import WorldModel
from ray.rllib.algorithms.dreamerv3.utils import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import inverse_symlog
def dream_trajectory_with_burn_in(self, *, start_states, timesteps_burn_in: int, timesteps_H: int, observations, actions, use_sampled_actions_in_dream: bool=False, use_random_actions_in_dream: bool=False):
    """Dreams trajectory from N initial observations and initial states.

        Note: This is only used for reporting and debugging, not for actual world-model
        or policy training.

        Args:
            start_states: The batch of start states (dicts with `a`, `h`, and `z` keys)
                to begin dreaming with. These are used to compute the first h-state
                using the sequence model.
            timesteps_burn_in: For how many timesteps should be use the posterior
                z-states (computed by the posterior net and actual observations from
                the env)?
            timesteps_H: For how many timesteps should we dream using the prior
                z-states (computed by the dynamics (prior) net and h-states only)?
                Note that the total length of the returned trajectories will
                be `timesteps_burn_in` + `timesteps_H`.
            observations: The batch (B, T, ...) of observations (to be used only during
                burn-in over `timesteps_burn_in` timesteps).
            actions: The batch (B, T, ...) of actions to use during a) burn-in over the
                first `timesteps_burn_in` timesteps and - possibly - b) during
                actual dreaming, iff use_sampled_actions_in_dream=True.
                If applicable, actions must already be one-hot'd.
            use_sampled_actions_in_dream: If True, instead of using our actor network
                to compute fresh actions, we will use the one provided via the `actions`
                argument. Note that in the latter case, the `actions` time dimension
                must be at least `timesteps_burn_in` + `timesteps_H` long.
            use_random_actions_in_dream: Whether to use randomly sampled actions in the
                dream. Note that this does not apply to the burn-in phase, during which
                we will always use the actions given in the `actions` argument.
        """
    assert not (use_sampled_actions_in_dream and use_random_actions_in_dream)
    B = observations.shape[0]
    states = start_states
    for i in range(timesteps_burn_in):
        states = self.world_model.forward_inference(observations=observations[:, i], previous_states=states, is_first=tf.fill((B,), 1.0 if i == 0 else 0.0))
        states['a'] = actions[:, i]
    h_states_t0_to_H = [states['h']]
    z_states_prior_t0_to_H = [states['z']]
    a_t0_to_H = [states['a']]
    for j in range(timesteps_H):
        h = self.world_model.sequence_model(a=states['a'], h=states['h'], z=states['z'])
        h_states_t0_to_H.append(h)
        z, _ = self.world_model.dynamics_predictor(h=h)
        z_states_prior_t0_to_H.append(z)
        if use_sampled_actions_in_dream:
            a = actions[:, timesteps_burn_in + j]
        elif use_random_actions_in_dream:
            if isinstance(self.action_space, gym.spaces.Discrete):
                a = tf.random.randint((B,), 0, self.action_space.n, tf.int64)
                a = tf.one_hot(a, depth=self.action_space.n, dtype=tf.keras.mixed_precision.global_policy().compute_dtype or tf.float32)
            else:
                a = tf.random.uniform(shape=(B,) + self.action_space.shape, dtype=self.action_space.dtype)
        else:
            a, _ = self.actor(h=h, z=z)
        a_t0_to_H.append(a)
        states = {'h': h, 'z': z, 'a': a}
    h_states_t0_to_H_B = tf.stack(h_states_t0_to_H, axis=0)
    h_states_t0_to_HxB = tf.reshape(h_states_t0_to_H_B, shape=[-1] + h_states_t0_to_H_B.shape.as_list()[2:])
    z_states_prior_t0_to_H_B = tf.stack(z_states_prior_t0_to_H, axis=0)
    z_states_prior_t0_to_HxB = tf.reshape(z_states_prior_t0_to_H_B, shape=[-1] + z_states_prior_t0_to_H_B.shape.as_list()[2:])
    a_t0_to_H_B = tf.stack(a_t0_to_H, axis=0)
    o_dreamed_t0_to_HxB = self.world_model.decoder(h=h_states_t0_to_HxB, z=z_states_prior_t0_to_HxB)
    if self.world_model.symlog_obs:
        o_dreamed_t0_to_HxB = inverse_symlog(o_dreamed_t0_to_HxB)
    r_dreamed_t0_to_HxB, _ = self.world_model.reward_predictor(h=h_states_t0_to_HxB, z=z_states_prior_t0_to_HxB)
    r_dreamed_t0_to_HxB = inverse_symlog(r_dreamed_t0_to_HxB)
    c_dreamed_t0_to_HxB, _ = self.world_model.continue_predictor(h=h_states_t0_to_HxB, z=z_states_prior_t0_to_HxB)
    ret = {'h_states_t0_to_H_BxT': h_states_t0_to_H_B, 'z_states_prior_t0_to_H_BxT': z_states_prior_t0_to_H_B, 'observations_dreamed_t0_to_H_BxT': tf.reshape(o_dreamed_t0_to_HxB, [-1, B] + list(observations.shape)[2:]), 'rewards_dreamed_t0_to_H_BxT': tf.reshape(r_dreamed_t0_to_HxB, (-1, B)), 'continues_dreamed_t0_to_H_BxT': tf.reshape(c_dreamed_t0_to_HxB, (-1, B))}
    if use_sampled_actions_in_dream:
        key = 'actions_sampled_t0_to_H_BxT'
    elif use_random_actions_in_dream:
        key = 'actions_random_t0_to_H_BxT'
    else:
        key = 'actions_dreamed_t0_to_H_BxT'
    ret[key] = a_t0_to_H_B
    if isinstance(self.action_space, gym.spaces.Discrete):
        ret[re.sub('^actions_', 'actions_ints_', key)] = tf.argmax(a_t0_to_H_B, axis=-1)
    return ret