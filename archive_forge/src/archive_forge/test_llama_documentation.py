import pytest
import ray
import vllm
from vllm.lora.request import LoRARequest
from .conftest import cleanup
Test that the LLM initialization works with a warmup LORA path and is more conservative