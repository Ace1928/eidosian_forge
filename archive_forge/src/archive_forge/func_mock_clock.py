from __future__ import annotations
import inspect
from typing import NoReturn
import pytest
from ..testing import MockClock, trio_test
@pytest.fixture
def mock_clock() -> MockClock:
    return MockClock()