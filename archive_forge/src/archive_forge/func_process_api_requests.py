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
def process_api_requests(lc_model, requests: Optional[List[Union[Any, Dict[str, Any]]]]=None, max_workers: int=10, callback_handlers: Optional[List[BaseCallbackHandler]]=None, convert_chat_responses: bool=False):
    """
    Processes API requests in parallel.
    """
    retry_queue = queue.Queue()
    status_tracker = StatusTracker()
    next_request = None
    results = []
    errors = {}
    converted_chat_requests, did_perform_chat_conversion = APIRequest._transform_request_json_for_chat_if_necessary(requests, lc_model)
    requests_iter = enumerate(converted_chat_requests)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            if not retry_queue.empty():
                next_request = retry_queue.get_nowait()
                _logger.warning(f'Retrying request {next_request.index}: {next_request}')
            elif (req := next(requests_iter, None)):
                index, converted_chat_request_json = req
                next_request = APIRequest(index=index, lc_model=lc_model, request_json=converted_chat_request_json, results=results, errors=errors, convert_chat_responses=convert_chat_responses, did_perform_chat_conversion=did_perform_chat_conversion, stream=False)
                status_tracker.start_task()
            else:
                next_request = None
            if next_request:
                executor.submit(next_request.call_api, status_tracker=status_tracker, callback_handlers=callback_handlers)
            if status_tracker.num_tasks_in_progress == 0 and next_request is None:
                break
            time.sleep(0.001)
        if status_tracker.num_tasks_failed > 0:
            raise mlflow.MlflowException(f'{status_tracker.num_tasks_failed} tasks failed. Errors: {errors}')
        return [res for _, res in sorted(results)]