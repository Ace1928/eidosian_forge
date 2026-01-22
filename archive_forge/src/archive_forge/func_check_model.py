import argparse
import asyncio
from http import HTTPStatus
import json
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import AsyncGenerator, Optional
from dataclasses import dataclass
import fastapi
from fastapi import BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
import uvicorn
import ray
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from ochat.config import MODEL_CONFIG_MAP
from ochat.serving import openai_api_protocol, async_tokenizer
from transformers.utils.hub import cached_file
def check_model(request) -> Optional[JSONResponse]:
    if request.model in model.names:
        return
    return create_error_response(HTTPStatus.NOT_FOUND, f'The model `{request.model}` does not exist.')