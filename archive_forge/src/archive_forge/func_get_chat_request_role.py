import time
import codecs
from fastapi import Request
from typing import AsyncGenerator, AsyncIterator, Optional, List, Union
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
from vllm.outputs import RequestOutput
from vllm.entrypoints.openai.serving_engine import OpenAIServing, LoRA
from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor
def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
    if request.add_generation_prompt:
        return self.response_role
    else:
        return request.messages[-1]['role']