import logging
from typing import Type, Dict, Any, Optional, Union
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.algorithms.sac.sac_tf_policy import SACTFPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
Sets the training related configuration.

        Args:
            twin_q: Use two Q-networks (instead of one) for action-value estimation.
                Note: Each Q-network will have its own target network.
            q_model_config: Model configs for the Q network(s). These will override
                MODEL_DEFAULTS. This is treated just as the top-level `model` dict in
                setting up the Q-network(s) (2 if twin_q=True).
                That means, you can do for different observation spaces:
                `obs=Box(1D)` -> `Tuple(Box(1D) + Action)` -> `concat` -> `post_fcnet`
                obs=Box(3D) -> Tuple(Box(3D) + Action) -> vision-net -> concat w/ action
                -> post_fcnet
                obs=Tuple(Box(1D), Box(3D)) -> Tuple(Box(1D), Box(3D), Action)
                -> vision-net -> concat w/ Box(1D) and action -> post_fcnet
                You can also have SAC use your custom_model as Q-model(s), by simply
                specifying the `custom_model` sub-key in below dict (just like you would
                do in the top-level `model` dict.
            policy_model_config: Model options for the policy function (see
                `q_model_config` above for details). The difference to `q_model_config`
                above is that no action concat'ing is performed before the post_fcnet
                stack.
            tau: Update the target by 	au * policy + (1-	au) * target_policy.
            initial_alpha: Initial value to use for the entropy weight alpha.
            target_entropy: Target entropy lower bound. If "auto", will be set
                to `-|A|` (e.g. -2.0 for Discrete(2), -3.0 for Box(shape=(3,))).
                This is the inverse of reward scale, and will be optimized
                automatically.
            n_step: N-step target updates. If >1, sars' tuples in trajectories will be
                postprocessed to become sa[discounted sum of R][s t+n] tuples.
            store_buffer_in_checkpoints: Set this to True, if you want the contents of
                your buffer(s) to be stored in any saved checkpoints as well.
                Warnings will be created if:
                - This is True AND restoring from a checkpoint that contains no buffer
                data.
                - This is False AND restoring from a checkpoint that does contain
                buffer data.
            replay_buffer_config: Replay buffer config.
                Examples:
                {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentReplayBuffer",
                "capacity": 50000,
                "replay_batch_size": 32,
                "replay_sequence_length": 1,
                }
                - OR -
                {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 50000,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
                "replay_sequence_length": 1,
                }
                - Where -
                prioritized_replay_alpha: Alpha parameter controls the degree of
                prioritization in the buffer. In other words, when a buffer sample has
                a higher temporal-difference error, with how much more probability
                should it drawn to use to update the parametrized Q-network. 0.0
                corresponds to uniform probability. Setting much above 1.0 may quickly
                result as the sampling distribution could become heavily “pointy” with
                low entropy.
                prioritized_replay_beta: Beta parameter controls the degree of
                importance sampling which suppresses the influence of gradient updates
                from samples that have higher probability of being sampled via alpha
                parameter and the temporal-difference error.
                prioritized_replay_eps: Epsilon parameter sets the baseline probability
                for sampling so that when the temporal-difference error of a sample is
                zero, there is still a chance of drawing the sample.
            training_intensity: The intensity with which to update the model (vs
                collecting samples from the env).
                If None, uses "natural" values of:
                `train_batch_size` / (`rollout_fragment_length` x `num_workers` x
                `num_envs_per_worker`).
                If not None, will make sure that the ratio between timesteps inserted
                into and sampled from th buffer matches the given values.
                Example:
                training_intensity=1000.0
                train_batch_size=250
                rollout_fragment_length=1
                num_workers=1 (or 0)
                num_envs_per_worker=1
                -> natural value = 250 / 1 = 250.0
                -> will make sure that replay+train op will be executed 4x asoften as
                rollout+insert op (4 * 250 = 1000).
                See: rllib/algorithms/dqn/dqn.py::calculate_rr_weights for further
                details.
            clip_actions: Whether to clip actions. If actions are already normalized,
                this should be set to False.
            grad_clip: If not None, clip gradients during optimization at this value.
            optimization_config: Config dict for optimization. Set the supported keys
                `actor_learning_rate`, `critic_learning_rate`, and
                `entropy_learning_rate` in here.
            target_network_update_freq: Update the target network every
                `target_network_update_freq` steps.
            _deterministic_loss: Whether the loss should be calculated deterministically
                (w/o the stochastic action sampling step). True only useful for
                continuous actions and for debugging.
            _use_beta_distribution: Use a Beta-distribution instead of a
                `SquashedGaussian` for bounded, continuous action spaces (not
                recommended; for debugging only).

        Returns:
            This updated AlgorithmConfig object.
        