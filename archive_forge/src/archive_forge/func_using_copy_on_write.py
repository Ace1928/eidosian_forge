import operator
import pytest
from pandas._config.config import _get_option
from pandas import (
@pytest.fixture
def using_copy_on_write() -> bool:
    """
    Fixture to check if Copy-on-Write is enabled.
    """
    return options.mode.copy_on_write is True and _get_option('mode.data_manager', silent=True) == 'block'