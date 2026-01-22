import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str
from langchain_core.utils.env import get_from_dict_or_env
def completion_with_retry_batching(llm: Fireworks, use_retry: bool, *, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    import fireworks.client
    prompt = kwargs['prompt']
    del kwargs['prompt']
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @conditional_decorator(use_retry, retry_decorator)
    def _completion_with_retry(prompt: str) -> Any:
        return fireworks.client.Completion.create(**kwargs, prompt=prompt)

    def batch_sync_run() -> List:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_completion_with_retry, prompt))
        return results
    return batch_sync_run()