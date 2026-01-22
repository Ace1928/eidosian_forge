import pytest
import ray
import vllm
from vllm.lora.request import LoRARequest
from .conftest import cleanup
def test_llama_lora_warmup(sql_lora_files):
    """Test that the LLM initialization works with a warmup LORA path and is more conservative"""

    @ray.remote(num_gpus=1)
    def get_num_gpu_blocks_lora():
        llm = vllm.LLM(MODEL_PATH, enable_lora=True, max_num_seqs=16)
        num_gpu_blocks_lora_warmup = llm.llm_engine.cache_config.num_gpu_blocks
        return num_gpu_blocks_lora_warmup

    @ray.remote(num_gpus=1)
    def get_num_gpu_blocks_no_lora():
        llm = vllm.LLM(MODEL_PATH, max_num_seqs=16)
        num_gpu_blocks_no_lora_warmup = llm.llm_engine.cache_config.num_gpu_blocks
        return num_gpu_blocks_no_lora_warmup
    num_gpu_blocks_lora_warmup = ray.get(get_num_gpu_blocks_lora.remote())
    num_gpu_blocks_no_lora_warmup = ray.get(get_num_gpu_blocks_no_lora.remote())
    assert num_gpu_blocks_lora_warmup < num_gpu_blocks_no_lora_warmup, 'The warmup with lora should be more conservative than without lora, therefore the number of memory blocks for the KV cache should be less when using lora than when not using lora'