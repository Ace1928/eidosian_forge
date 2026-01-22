import collections
import logging
from enum import Enum
from typing import Any, Dict, Optional
from ray.util.timer import _Timer
from ray.rllib.policy.rnn_sequencing import timeslice_along_seq_lens_with_overlap
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.replay_buffers.replay_buffer import (
from ray.rllib.utils.typing import PolicyID, SampleBatchType
from ray.util.annotations import DeveloperAPI
from ray.util.debug import log_once
Returns a dict of policy IDs and batches, depending on our replay mode.

        This method helps with splitting up MultiAgentBatches only if the
        self.replay_mode requires it.
        