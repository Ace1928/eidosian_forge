import argparse
import contextlib
import functools
import types
from typing import Any, Sequence, Text, TextIO, Tuple, Type, Optional, Union
from typing import Callable, ContextManager, Generator
import autopage
from argparse import *  # noqa
@contextlib.contextmanager
def unpatcher() -> Generator:
    try:
        yield
    finally:
        patch_classes(argparse, orig)
        argparse.ArgumentParser._get_formatter = orig_fmtr