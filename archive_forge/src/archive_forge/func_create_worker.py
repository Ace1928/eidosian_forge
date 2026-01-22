import torch
from typing import List, Optional, Dict
from vllm.worker.worker import Worker
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import SequenceGroupMetadata, SequenceData
from vllm.sampling_params import SamplingParams
from vllm.worker.cache_engine import CacheEngine
from vllm.model_executor.utils import set_random_seed
from dataclasses import dataclass, fields
def create_worker(cls: type, model_name: str, block_size: int, num_gpu_blocks: int, seed: int, is_driver_worker: bool=True, enforce_eager: bool=True):
    engine_args = EngineArgs(model=model_name, seed=seed, block_size=block_size, enforce_eager=enforce_eager)
    model_config, cache_config, parallel_config, scheduler_config, device_config, _ = engine_args.create_engine_configs()
    distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
    worker = cls(model_config=model_config, parallel_config=parallel_config, scheduler_config=scheduler_config, device_config=device_config, local_rank=0, rank=0, distributed_init_method=distributed_init_method, is_driver_worker=is_driver_worker)
    worker.init_model()
    worker.load_model()
    cache_config.num_gpu_blocks = num_gpu_blocks
    cache_config.num_cpu_blocks = 0
    worker.init_cache_engine(cache_config)
    worker.warm_up_model()
    return worker