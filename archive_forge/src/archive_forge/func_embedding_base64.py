import asyncio
import json
from threading import Lock
from typing import List, Union
from enum import Enum
import base64
from fastapi import APIRouter, Request, status, HTTPException
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
import tiktoken
from utils.rwkv import *
from utils.log import quick_log
import global_var
def embedding_base64(embedding: List[float]) -> str:
    import numpy as np
    return base64.b64encode(np.array(embedding).astype(np.float32)).decode('utf-8')