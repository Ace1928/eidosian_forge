from __future__ import annotations
import base64
import hashlib
import hmac
import json
import logging
import queue
import threading
from datetime import datetime
from queue import Queue
from time import mktime
from typing import Any, Dict, Generator, Iterator, List, Optional
from urllib.parse import urlencode, urlparse, urlunparse
from wsgiref.handlers import format_date_time
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def gen_params(self, messages: list, user_id: str, model_kwargs: Optional[dict]=None) -> dict:
    data: Dict = {'header': {'app_id': self.app_id, 'uid': user_id}, 'parameter': {'chat': {'domain': self.spark_domain}}, 'payload': {'message': {'text': messages}}}
    if model_kwargs:
        data['parameter']['chat'].update(model_kwargs)
    logger.debug(f'Spark Request Parameters: {data}')
    return data