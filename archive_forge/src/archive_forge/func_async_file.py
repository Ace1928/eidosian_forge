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
@pytest.fixture
def async_file(wrapped: mock.Mock) -> AsyncIOWrapper[mock.Mock]:
    return trio.wrap_file(wrapped)