from lazyops import lazy_init, get_logger
import asyncio
import inspect
import contextvars
from collections import ChainMap
from json import JSONDecodeError
from types import FunctionType, CoroutineType
from typing import List, Union, Any, Callable, Type, Optional, Dict, Sequence, Awaitable, Tuple
from contextlib import AsyncExitStack, AbstractAsyncContextManager, asynccontextmanager, contextmanager
from pydantic import DictError
from pydantic import StrictStr, ValidationError
from pydantic import BaseModel as PyBaseModel
from pydantic import BaseConfig
from pydantic.fields import ModelField, Field
from pydantic.main import ModelMetaclass
from fastapi.dependencies.models import Dependant
from fastapi.encoders import jsonable_encoder
from fastapi.params import Depends
from fastapi import FastAPI, Body, Header
from fastapi.dependencies.utils import solve_dependencies, get_dependant, get_flat_dependant, get_parameterless_sub_dependant
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.routing import APIRoute, APIRouter, serialize_response
from starlette.background import BackgroundTasks
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response
from starlette.responses import JSONResponse as BaseJSONResponse
from starlette.routing import Match, request_response, compile_path
import aiojobs
import fastapi.params
import simdjson as json
def sjson_dumps(v, *, default):
    return json.dumps(v, default=default).decode()