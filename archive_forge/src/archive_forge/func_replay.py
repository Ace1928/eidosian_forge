import collections
import platform
import random
from typing import Optional
from ray.util.timer import _Timer
from ray.rllib.execution.replay_ops import SimpleReplayBuffer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, concat_samples
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import ReplayMode
from ray.rllib.utils.replay_buffers.replay_buffer import _ALL_POLICIES
from ray.rllib.utils.typing import PolicyID, SampleBatchType
def replay(self, policy_id: PolicyID=DEFAULT_POLICY_ID) -> Optional[SampleBatchType]:
    if self.replay_mode == ReplayMode.LOCKSTEP and policy_id != _ALL_POLICIES:
        raise ValueError("Trying to sample from single policy's buffer in lockstep mode. In lockstep mode, all policies' experiences are sampled from a single replay buffer which is accessed with the policy id `{}`".format(_ALL_POLICIES))
    buffer = self.replay_buffers[policy_id]
    if len(buffer) == 0 or (len(self.last_added_batches[policy_id]) == 0 and self.replay_ratio < 1.0):
        return None
    with self.replay_timer:
        output_batches = self.last_added_batches[policy_id]
        self.last_added_batches[policy_id] = []
        if self.replay_ratio == 0.0:
            return concat_samples(output_batches)
        elif self.replay_ratio == 1.0:
            return buffer.replay()
        num_new = len(output_batches)
        replay_proportion = self.replay_proportion
        while random.random() < num_new * replay_proportion:
            replay_proportion -= 1
            output_batches.append(buffer.replay())
        return concat_samples(output_batches)