import copy
from collections import defaultdict
import os
import time
import pickle
import importlib
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple,
from vllm.lora.request import LoRARequest
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import StatLogger, Stats
from vllm.engine.ray_utils import RayWorkerVllm, initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup,
from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
from vllm.utils import (Counter, set_cuda_visible_devices, get_ip,
@classmethod
def from_engine_args(cls, engine_args: EngineArgs) -> 'LLMEngine':
    """Creates an LLM engine from the engine arguments."""
    engine_configs = engine_args.create_engine_configs()
    parallel_config = engine_configs[2]
    placement_group = initialize_cluster(parallel_config)
    engine = cls(*engine_configs, placement_group, log_stats=not engine_args.disable_log_stats)
    return engine