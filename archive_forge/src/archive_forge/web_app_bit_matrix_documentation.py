import asyncio
import logging
import logging.config
from typing import List, Tuple, Optional, Union, Any
import functools
from functools import wraps
import cProfile
import pstats
import io
import tracemalloc
import signal
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from memory_profiler import profile
import cachetools.func
from cachetools import TTLCache
import aiofiles
import aiohttp
from aiohttp import web
import json
from datetime import datetime
from sympy import true

    Main function to drive the program.

    Sets up the web server routes and handles the main logic for constructing the bit matrix.
    