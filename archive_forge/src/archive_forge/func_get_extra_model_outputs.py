from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def get_extra_model_outputs(self, indices: Union[int, List[int]]=-1, global_ts: bool=True, as_list: bool=False) -> Union[MultiAgentDict, List[MultiAgentDict]]:
    """Gets actions for all agents that stepped in the last timesteps.

        Note that actions are only returned for agents that stepped
        during the given index range.

        Args:
            indices: Either a single index or a list of indices. The indices
                can be reversed (e.g. [-1, -2]) or absolute (e.g. [98, 99]).
                This defines the time indices for which the actions
                should be returned.
            global_ts: Boolean that defines, if the indices should be considered
                environment (`True`) or agent (`False`) steps.

        Returns: A dictionary mapping agent ids to actions (of different
            timesteps). Only for agents that have stepped (were ready) at a
            timestep, actions are returned (i.e. not all agent ids are
            necessarily in the keys).
        """
    buffered_outputs = {}
    if global_ts:
        if isinstance(indices, list):
            indices = [self.t + idx + 1 if idx < 0 else idx for idx in indices]
        else:
            indices = [self.t + indices + 1] if indices < 0 else [indices]
    elif not isinstance(indices, list):
        indices = [indices]
    for agent_id, agent_global_action_t in self.global_actions_t.items():
        if agent_global_action_t:
            last_action_index = agent_global_action_t[-1] if global_ts else len(agent_global_action_t) - 1
        if agent_global_action_t and (last_action_index in indices or -1 in indices) and self.agent_buffers[agent_id]['actions'].full():
            buffered_outputs[agent_id] = [self.agent_buffers[agent_id]['extra_model_outputs'].queue[0]]
        else:
            buffered_outputs[agent_id] = []
    extra_model_outputs = self._getattr_by_index('extra_model_outputs', indices=indices, has_initial_value=True, global_ts=global_ts, global_ts_mapping=self.global_actions_t, as_list=as_list, buffered_values=buffered_outputs)
    return extra_model_outputs