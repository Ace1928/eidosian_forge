from typing import Any, AsyncIterator, Iterator, List, Optional
from langchain_core.callbacks.manager import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_core.pydantic_v1 import root_validator
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.llms.huggingface_text_gen_inference import (
@root_validator()
def validate_llm(cls, values: dict) -> dict:
    if not isinstance(values['llm'], (HuggingFaceTextGenInference, HuggingFaceEndpoint, HuggingFaceHub)):
        raise TypeError(f'Expected llm to be one of HuggingFaceTextGenInference, HuggingFaceEndpoint, HuggingFaceHub, received {type(values['llm'])}')
    return values