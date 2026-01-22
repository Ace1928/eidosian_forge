import logging
import os
import shutil
import tempfile
from typing import Any, Dict
import torch
from packaging.version import Version
import ray
from ray import train
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train import Checkpoint
from ray.util import PublicAPI
@property
def distributed_sampler_kwargs(self) -> Dict[str, Any]:
    return dict(num_replicas=self.world_size, rank=self.global_rank)