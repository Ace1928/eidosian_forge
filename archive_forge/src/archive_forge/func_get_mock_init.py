import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
def get_mock_init(name):

    def mock_init(obj, *args, **kwargs):
        pytest.skip(f'use --calculators={name} to enable')
    return mock_init