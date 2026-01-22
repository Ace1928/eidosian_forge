import inspect
import os
import sys
import traceback
from datetime import datetime
from enum import Enum
from functools import update_wrapper
from pathlib import Path
from traceback import FrameSummary, StackSummary
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from uuid import UUID
import click
from .completion import get_completion_inspect_parameters
from .core import MarkupMode, TyperArgument, TyperCommand, TyperGroup, TyperOption
from .models import (
from .utils import get_params_from_function
def generate_list_convertor(convertor: Optional[Callable[[Any], Any]], default_value: Optional[Any]) -> Callable[[Sequence[Any]], Optional[List[Any]]]:

    def internal_convertor(value: Sequence[Any]) -> Optional[List[Any]]:
        if default_value is None and len(value) == 0:
            return None
        return [convertor(v) if convertor else v for v in value]
    return internal_convertor