import gc
import os
import platform
import tracemalloc
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.deprecation import Deprecated, deprecation_warning
from ray.rllib.utils.exploration.random_encoder import (
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from ray.tune.callback import _CallbackMeta
import psutil
def make_multi_callbacks(callback_class_list: List[Type[DefaultCallbacks]]) -> DefaultCallbacks:
    """Allows combining multiple sub-callbacks into one new callbacks class.

    The resulting DefaultCallbacks will call all the sub-callbacks' callbacks
    when called.

    .. testcode::
        :skipif: True

        config.callbacks(make_multi_callbacks([
            MyCustomStatsCallbacks,
            MyCustomVideoCallbacks,
            MyCustomTraceCallbacks,
            ....
        ]))

    Args:
        callback_class_list: The list of sub-classes of DefaultCallbacks to
            be baked into the to-be-returned class. All of these sub-classes'
            implemented methods will be called in the given order.

    Returns:
        A DefaultCallbacks subclass that combines all the given sub-classes.
    """

    class _MultiCallbacks(DefaultCallbacks):
        IS_CALLBACK_CONTAINER = True

        def __init__(self):
            super().__init__()
            self._callback_list = [callback_class() for callback_class in callback_class_list]

        @override(DefaultCallbacks)
        def on_algorithm_init(self, *, algorithm: 'Algorithm', **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_algorithm_init(algorithm=algorithm, **kwargs)

        @override(DefaultCallbacks)
        def on_workers_recreated(self, **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_workers_recreated(**kwargs)

        @override(DefaultCallbacks)
        def on_checkpoint_loaded(self, *, algorithm: 'Algorithm', **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_checkpoint_loaded(algorithm=algorithm, **kwargs)

        @override(DefaultCallbacks)
        def on_create_policy(self, *, policy_id: PolicyID, policy: Policy) -> None:
            for callback in self._callback_list:
                callback.on_create_policy(policy_id=policy_id, policy=policy)

        @override(DefaultCallbacks)
        def on_sub_environment_created(self, *, worker: 'RolloutWorker', sub_environment: EnvType, env_context: EnvContext, env_index: Optional[int]=None, **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_sub_environment_created(worker=worker, sub_environment=sub_environment, env_context=env_context, **kwargs)

        @override(DefaultCallbacks)
        def on_episode_created(self, *, worker: 'RolloutWorker', base_env: BaseEnv, policies: Dict[PolicyID, Policy], env_index: int, episode: Union[Episode, EpisodeV2], **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_episode_created(worker=worker, base_env=base_env, policies=policies, env_index=env_index, episode=episode, **kwargs)

        @override(DefaultCallbacks)
        def on_episode_start(self, *, worker: 'RolloutWorker', base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Union[Episode, EpisodeV2], env_index: Optional[int]=None, **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode, env_index=env_index, **kwargs)

        @override(DefaultCallbacks)
        def on_episode_step(self, *, worker: 'RolloutWorker', base_env: BaseEnv, policies: Optional[Dict[PolicyID, Policy]]=None, episode: Union[Episode, EpisodeV2], env_index: Optional[int]=None, **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_episode_step(worker=worker, base_env=base_env, policies=policies, episode=episode, env_index=env_index, **kwargs)

        @override(DefaultCallbacks)
        def on_episode_end(self, *, worker: 'RolloutWorker', base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Union[Episode, EpisodeV2, Exception], env_index: Optional[int]=None, **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_episode_end(worker=worker, base_env=base_env, policies=policies, episode=episode, env_index=env_index, **kwargs)

        @override(DefaultCallbacks)
        def on_evaluate_start(self, *, algorithm: 'Algorithm', **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_evaluate_start(algorithm=algorithm, **kwargs)

        @override(DefaultCallbacks)
        def on_evaluate_end(self, *, algorithm: 'Algorithm', evaluation_metrics: dict, **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_evaluate_end(algorithm=algorithm, evaluation_metrics=evaluation_metrics, **kwargs)

        @override(DefaultCallbacks)
        def on_postprocess_trajectory(self, *, worker: 'RolloutWorker', episode: Episode, agent_id: AgentID, policy_id: PolicyID, policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch, original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]], **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_postprocess_trajectory(worker=worker, episode=episode, agent_id=agent_id, policy_id=policy_id, policies=policies, postprocessed_batch=postprocessed_batch, original_batches=original_batches, **kwargs)

        @override(DefaultCallbacks)
        def on_sample_end(self, *, worker: 'RolloutWorker', samples: SampleBatch, **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_sample_end(worker=worker, samples=samples, **kwargs)

        @override(DefaultCallbacks)
        def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_learn_on_batch(policy=policy, train_batch=train_batch, result=result, **kwargs)

        @override(DefaultCallbacks)
        def on_train_result(self, *, algorithm=None, result: dict, **kwargs) -> None:
            for callback in self._callback_list:
                callback.on_train_result(algorithm=algorithm, result=result, **kwargs)
    return _MultiCallbacks