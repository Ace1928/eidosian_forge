import asyncio
import time
from functools import partial
from typing import (Any, Dict, Iterable, List, Optional, Set, Tuple, Type,
from vllm.lora.request import LoRARequest
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
def start_background_loop(self) -> None:
    """Start the background loop."""
    if self.is_running:
        raise RuntimeError('Background loop is already running.')
    self._request_tracker.init_event()
    self._background_loop_unshielded = asyncio.get_event_loop().create_task(self.run_engine_loop())
    self._background_loop_unshielded.add_done_callback(partial(_raise_exception_on_finish, request_tracker=self._request_tracker))
    self.background_loop = asyncio.shield(self._background_loop_unshielded)