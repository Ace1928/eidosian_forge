from __future__ import annotations
import errno
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING
@contextmanager
def cd(path: Union[str, Path]) -> Generator:
    """
    A Fabric-inspired cd context that temporarily changes directory for
    performing some tasks, and returns to the original working directory
    afterwards. E.g.,

        with cd("/my/path/"):
            do_something()

    Args:
        path: Path to cd to.
    """
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)