from __future__ import annotations
import importlib
import io
import os
import re
from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import sentinel
import pytest
import trio
from trio import _core, _file_io
from trio._file_io import _FILE_ASYNC_METHODS, _FILE_SYNC_ATTRS, AsyncIOWrapper
def test_async_methods_signature(async_file: AsyncIOWrapper[mock.Mock]) -> None:
    assert async_file.read.__name__ == 'read'
    assert async_file.read.__qualname__ == 'AsyncIOWrapper.read'
    assert async_file.read.__doc__ is not None
    assert 'io.StringIO.read' in async_file.read.__doc__