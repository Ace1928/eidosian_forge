import os
from typing import Optional, TYPE_CHECKING
from ray.rllib.utils.annotations import PublicAPI
@PublicAPI
def default_sampler_input(self) -> Optional['SamplerInput']:
    """Returns the RolloutWorker's SamplerInput object, if any.

        Returns None if the RolloutWorker has no SamplerInput. Note that local
        workers in case there are also one or more remote workers by default
        do not create a SamplerInput object.

        Returns:
            The RolloutWorkers' SamplerInput object or None if none exists.
        """
    return self.worker.sampler