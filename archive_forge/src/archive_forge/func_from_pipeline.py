import importlib.util
import logging
import pickle
from typing import Any, Callable, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra
from langchain_community.llms.utils import enforce_stop_tokens
@classmethod
def from_pipeline(cls, pipeline: Any, hardware: Any, model_reqs: Optional[List[str]]=None, device: int=0, **kwargs: Any) -> LLM:
    """Init the SelfHostedPipeline from a pipeline object or string."""
    if not isinstance(pipeline, str):
        logger.warning('Serializing pipeline to send to remote hardware. Note, it can be quite slowto serialize and send large models with each execution. Consider sending the pipelineto the cluster and passing the path to the pipeline instead.')
    load_fn_kwargs = {'pipeline': pipeline, 'device': device}
    return cls(load_fn_kwargs=load_fn_kwargs, model_load_fn=_send_pipeline_to_device, hardware=hardware, model_reqs=['transformers', 'torch'] + (model_reqs or []), **kwargs)