import datetime as dt
import logging
import platform
import threading
import time
import uuid
from enum import IntEnum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
import pandas
import psutil
import modin
from modin.config import LogFileSize, LogMemoryInterval, LogMode
def memory_thread(logger: logging.Logger, sleep_time: int) -> None:
    """
    Configure Modin logging system memory profiling thread.

    Parameters
    ----------
    logger : logging.Logger
        The logger object.
    sleep_time : int
        The interval at which to profile system memory.
    """
    while True:
        rss_mem = bytes_int_to_str(psutil.Process().memory_info().rss)
        svmem = psutil.virtual_memory()
        logger.info(f'Memory Percentage: {svmem.percent}%')
        logger.info(f'RSS Memory: {rss_mem}')
        time.sleep(sleep_time)