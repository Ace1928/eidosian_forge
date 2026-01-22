import abc
import time
import random
import asyncio
import contextlib
from pydantic import BaseModel
from lazyops.imports._aiohttpx import resolve_aiohttpx
from lazyops.imports._bs4 import resolve_bs4
from urllib.parse import quote_plus
import aiohttpx
from bs4 import BeautifulSoup, Tag
from .utils import filter_result, load_user_agents, load_cookie_jar, get_random_jitter
from typing import List, Optional, Dict, Any, Union, Tuple, Set, Callable, Awaitable, TypeVar, Generator, AsyncGenerator
def validate_extra_params(self, extra_params: Dict[str, str]):
    """
        Validates the extra params
        """
    for builtin_param in url_parameters:
        if builtin_param in extra_params:
            raise ValueError('GET parameter "%s" is overlapping with                     the built-in GET parameter', builtin_param)