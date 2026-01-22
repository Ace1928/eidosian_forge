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
def fix_table_name(table: str) -> str:
    """Add single quotes around table names that contain spaces."""
    if ' ' in table and (not table.startswith("'")) and (not table.endswith("'")):
        return f"'{table}'"
    return table