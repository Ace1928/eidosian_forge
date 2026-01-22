import logging
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EpisodeID, PolicyID, TensorType
@abstractmethod
def postprocess_episode(self, episode: Episode, is_done: bool=False, check_dones: bool=False, build: bool=False) -> Optional[MultiAgentBatch]:
    """Postprocesses all agents' trajectories in a given episode.

        Generates (single-trajectory) SampleBatches for all Policies/Agents and
        calls Policy.postprocess_trajectory on each of these. Postprocessing
        may happens in-place, meaning any changes to the viewed data columns
        are directly reflected inside this collector's buffers.
        Also makes sure that additional (newly created) data columns are
        correctly added to the buffers.

        Args:
            episode: The Episode object for which
                to post-process data.
            is_done: Whether the given episode is actually terminated
                (all agents are terminated OR truncated). If True, the
                episode will no longer be used/continued and we may need to
                recycle/erase it internally. If a soft-horizon is hit, the
                episode will continue to be used and `is_done` should be set
                to False here.
            check_dones: Whether we need to check that all agents'
                trajectories have dones=True at the end.
            build: Whether to build a MultiAgentBatch from the given
                episode (and only that episode!) and return that
                MultiAgentBatch. Used for batch_mode=`complete_episodes`.

        Returns:
            Optional[MultiAgentBatch]: If `build` is True, the
                SampleBatch or MultiAgentBatch built from `episode` (either
                just from that episde or from the `_PolicyCollectorGroup`
                in the `episode.batch_builder` property).
        """
    raise NotImplementedError