from __future__ import annotations
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import requests
import tiktoken
import mlflow
from mlflow.protos.databricks_pb2 import UNAUTHENTICATED
from mlflow.utils.openai_utils import _OAITokenHolder
def num_tokens_consumed_from_request(request_json: dict, request_url: str, token_encoding_name: str):
    """
    Count the number of tokens in the request. Only supports completion and embedding requests.
    """
    encoding = tiktoken.get_encoding(token_encoding_name)
    if 'completions' in request_url:
        max_tokens = request_json.get('max_tokens', 15)
        n = request_json.get('n', 1)
        completion_tokens = n * max_tokens
        if 'chat/completions' in request_url:
            num_tokens = 0
            for message in request_json['messages']:
                num_tokens += 4
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == 'name':
                        num_tokens -= 1
            num_tokens += 2
            return num_tokens + completion_tokens
        else:
            prompt = request_json['prompt']
            if isinstance(prompt, str):
                prompt_tokens = len(encoding.encode(prompt))
                return prompt_tokens + completion_tokens
            elif isinstance(prompt, list):
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                return prompt_tokens + completion_tokens * len(prompt)
            else:
                raise TypeError("Expecting either string or list of strings for 'prompt' field in completion request")
    elif 'embeddings' in request_url:
        inp = request_json['input']
        if isinstance(inp, str):
            return len(encoding.encode(inp))
        elif isinstance(inp, list):
            return sum([len(encoding.encode(i)) for i in inp])
        else:
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    else:
        raise NotImplementedError(f'Support for "{request_url}" not implemented in this script')