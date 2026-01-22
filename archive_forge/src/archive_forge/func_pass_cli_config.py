import importlib
import os
import sys
import time
from ast import literal_eval
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import partial, update_wrapper
from json import JSONDecodeError, loads
from shutil import get_terminal_size
import click
from redis import Redis
from redis.sentinel import Sentinel
from rq.defaults import (
from rq.logutils import setup_loghandlers
from rq.utils import import_attribute, parse_timeout
from rq.worker import WorkerStatus
def pass_cli_config(func):
    for option in shared_options:
        func = option(func)

    def wrapper(*args, **kwargs):
        ctx = click.get_current_context()
        cli_config = CliConfig(**kwargs)
        return ctx.invoke(func, cli_config, *args[1:], **kwargs)
    return update_wrapper(wrapper, func)