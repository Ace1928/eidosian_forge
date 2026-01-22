from __future__ import annotations
import asyncio
import contextlib
import contextvars
import datetime
import functools
import inspect
import logging
import traceback
import uuid
import warnings
from contextvars import copy_context
from typing import (
from langsmith import client as ls_client
from langsmith import run_trees, utils
from langsmith._internal import _aiter as aitertools
def wrap_traceable(inputs: dict, config: RunnableConfig) -> Any:
    run_tree = RunnableTraceable._configure_run_tree(config.get('callbacks'))
    return func(**inputs, langsmith_extra={'run_tree': run_tree})