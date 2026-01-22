from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import SampleBatchType
Saves a batch of experiences.

        Args:
            sample_batch: SampleBatch or MultiAgentBatch to save.
        