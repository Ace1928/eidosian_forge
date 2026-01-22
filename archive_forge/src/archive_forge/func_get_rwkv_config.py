from abc import ABC, abstractmethod
from enum import Enum, auto
import os
import pathlib
import copy
import re
from typing import Dict, Iterable, List, Tuple, Union, Type, Callable
from utils.log import quick_log
from fastapi import HTTPException
from pydantic import BaseModel, Field
from routes import state_cache
import global_var
def get_rwkv_config(model: AbstractRWKV) -> ModelConfigBody:
    return ModelConfigBody(max_tokens=model.max_tokens_per_generation, temperature=model.temperature, top_p=model.top_p, presence_penalty=model.penalty_alpha_presence, frequency_penalty=model.penalty_alpha_frequency, penalty_decay=model.penalty_decay, top_k=model.top_k, global_penalty=model.global_penalty)