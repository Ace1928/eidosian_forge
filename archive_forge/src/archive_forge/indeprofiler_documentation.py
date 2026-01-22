import asyncio
import logging
import tracemalloc
import traceback
import memory_profiler
import cProfile
import pstats
import psutil
import aiofiles  # Correcting the missing import based on lint_context_0
from typing_extensions import NoReturn
from typing import Optional, Literal, Union
from scripts.trading_bot.indehandler import (

        Asynchronously logs the current resource usage metrics, including CPU, memory, and traced memory.
        This method gathers resource usage metrics and writes them to both the specified output file and the log.
        