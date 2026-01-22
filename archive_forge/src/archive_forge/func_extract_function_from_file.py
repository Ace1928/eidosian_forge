import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union
import requests
from outlines import generate, models
def extract_function_from_file(content: str, function_name: str) -> Tuple[Callable]:
    """Extract a function object from a downloaded file."""
    spec = importlib.util.spec_from_loader('outlines_function', loader=None, origin='github')
    if spec is not None:
        module = importlib.util.module_from_spec(spec)
        exec(content, module.__dict__)
        try:
            fn = getattr(module, function_name)
        except AttributeError:
            raise AttributeError('Could not find an `outlines.Function` instance in the remote file. Make sure that the path you specified is correct.')
        if not isinstance(fn, module.outlines.Function):
            raise TypeError(f'The `{function_name}` variable in the program must be an instance of `outlines.Function`')
    return fn