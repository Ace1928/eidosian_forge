from __future__ import annotations
import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union
import aiohttp
import requests
from aiohttp import ServerTimeoutError
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator, validator
from requests.exceptions import Timeout
@validator('table_names', allow_reuse=True)
def fix_table_names(cls, table_names: List[str]) -> List[str]:
    """Fix the table names."""
    return [fix_table_name(table) for table in table_names]