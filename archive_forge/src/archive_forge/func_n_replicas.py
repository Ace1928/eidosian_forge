import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple
from .training_args import TrainingArguments
from .utils import cached_property, is_tf_available, logging, requires_backends
@property
def n_replicas(self) -> int:
    """
        The number of replicas (CPUs, GPUs or TPU cores) used in this training.
        """
    requires_backends(self, ['tf'])
    return self._setup_strategy.num_replicas_in_sync