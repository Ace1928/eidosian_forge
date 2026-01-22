import inspect
import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
import click
import colorama
import ray  # noqa: F401
def indented(self):
    """Context manager that starts an indented block of output."""
    cli_logger = self

    class IndentedContextManager:

        def __enter__(self):
            cli_logger.indent_level += 1

        def __exit__(self, type, value, tb):
            cli_logger.indent_level -= 1
    return IndentedContextManager()