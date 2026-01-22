from typing import Dict, List, Optional, Tuple
import torch
import torch.distributed
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner
def init_cache_engine(self, cache_config: CacheConfig) -> None:
    self.cache_config = cache_config
    self.cache_engine = CacheEngine(self.cache_config, self.model_config, self.parallel_config)
    self.model_runner.set_block_size(self.cache_engine.block_size)