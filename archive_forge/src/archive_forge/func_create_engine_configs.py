import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
def create_engine_configs(self) -> Tuple[ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig, DeviceConfig, Optional[LoRAConfig]]:
    device_config = DeviceConfig(self.device)
    model_config = ModelConfig(self.model, self.tokenizer, self.tokenizer_mode, self.trust_remote_code, self.download_dir, self.load_format, self.dtype, self.seed, self.revision, self.code_revision, self.tokenizer_revision, self.max_model_len, self.quantization, self.enforce_eager, self.max_context_len_to_capture)
    cache_config = CacheConfig(self.block_size, self.gpu_memory_utilization, self.swap_space, self.kv_cache_dtype, model_config.get_sliding_window())
    parallel_config = ParallelConfig(self.pipeline_parallel_size, self.tensor_parallel_size, self.worker_use_ray, self.max_parallel_loading_workers, self.disable_custom_all_reduce)
    scheduler_config = SchedulerConfig(self.max_num_batched_tokens, self.max_num_seqs, model_config.max_model_len, self.max_paddings)
    lora_config = LoRAConfig(max_lora_rank=self.max_lora_rank, max_loras=self.max_loras, lora_extra_vocab_size=self.lora_extra_vocab_size, lora_dtype=self.lora_dtype, max_cpu_loras=self.max_cpu_loras if self.max_cpu_loras and self.max_cpu_loras > 0 else None) if self.enable_lora else None
    return (model_config, cache_config, parallel_config, scheduler_config, device_config, lora_config)