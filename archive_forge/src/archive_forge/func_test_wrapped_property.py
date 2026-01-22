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
def test_wrapped_property(async_file: AsyncIOWrapper[mock.Mock], wrapped: mock.Mock) -> None:
    assert async_file.wrapped is wrapped