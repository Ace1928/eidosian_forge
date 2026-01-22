from __future__ import annotations
from contextlib import contextmanager
from typing import Any
from streamlit import config
@contextmanager
def patch_config_options(config_overrides: dict[str, Any]):
    """A context manager that overrides config options. It can
    also be used as a function decorator.

    Examples:
    >>> with patch_config_options({"server.headless": True}):
    ...     assert(config.get_option("server.headless") is True)
    ...     # Other test code that relies on these options

    >>> @patch_config_options({"server.headless": True})
    ... def test_my_thing():
    ...   assert(config.get_option("server.headless") is True)
    """
    from unittest.mock import patch
    mock_get_option = build_mock_config_get_option(config_overrides)
    with patch.object(config, 'get_option', new=mock_get_option):
        yield