import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union
import requests
from outlines import generate, models
@classmethod
def from_github(cls, program_path: str, function_name: str='fn'):
    """Load a function stored on GitHub"""
    program_content = download_from_github(program_path)
    function = extract_function_from_file(program_content, function_name)
    return function