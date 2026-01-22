import os
import sys
from typing import Any, Dict, Optional, Tuple
import aiohttp
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
Process response from SerpAPI.