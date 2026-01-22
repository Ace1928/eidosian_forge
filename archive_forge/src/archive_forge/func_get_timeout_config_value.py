import os
import sys
from typing import Generator
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
from _pytest.stash import StashKey
import pytest
def get_timeout_config_value(config: Config) -> float:
    return float(config.getini('faulthandler_timeout') or 0.0)