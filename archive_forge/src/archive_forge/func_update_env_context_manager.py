import contextlib
import hashlib
import logging
import os
from types import TracebackType
from typing import Dict, Generator, Optional, Set, Type, Union
from pip._internal.models.link import Link
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.temp_dir import TempDirectory
@contextlib.contextmanager
def update_env_context_manager(**changes: str) -> Generator[None, None, None]:
    target = os.environ
    non_existent_marker = object()
    saved_values: Dict[str, Union[object, str]] = {}
    for name, new_value in changes.items():
        try:
            saved_values[name] = target[name]
        except KeyError:
            saved_values[name] = non_existent_marker
        target[name] = new_value
    try:
        yield
    finally:
        for name, original_value in saved_values.items():
            if original_value is non_existent_marker:
                del target[name]
            else:
                assert isinstance(original_value, str)
                target[name] = original_value