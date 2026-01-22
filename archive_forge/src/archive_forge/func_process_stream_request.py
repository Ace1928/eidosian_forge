from __future__ import annotations
import json
import logging
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Union
import langchain.chains
import pydantic
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AIMessage, HumanMessage, SystemMessage
from langchain.schema import ChatMessage as LangChainChatMessage
from packaging.version import Version
import mlflow
from mlflow.exceptions import MlflowException
def process_stream_request(lc_model, request_json: Union[Any, Dict[str, Any]], callback_handlers: Optional[List[BaseCallbackHandler]]=None, convert_chat_responses: bool=False):
    from mlflow.langchain.utils import lc_runnables_types
    if not isinstance(lc_model, lc_runnables_types()):
        raise MlflowException(f'Model {lc_model.__class__.__name__} does not support streaming prediction output.')
    converted_chat_requests, did_perform_chat_conversion = APIRequest._transform_request_json_for_chat_if_necessary(request_json, lc_model)
    api_request = APIRequest(index=0, lc_model=lc_model, request_json=converted_chat_requests, results=None, errors=None, convert_chat_responses=convert_chat_responses, did_perform_chat_conversion=did_perform_chat_conversion, stream=True)
    return api_request.single_call_api(callback_handlers)