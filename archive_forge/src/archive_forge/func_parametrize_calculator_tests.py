import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
def parametrize_calculator_tests(metafunc):
    """Parametrize tests using our custom markers.

    We want tests marked with @pytest.mark.calculator(names) to be
    parametrized over the named calculator or calculators."""
    calculator_inputs = []
    for marker in metafunc.definition.iter_markers(name='calculator'):
        calculator_names = marker.args
        kwargs = dict(marker.kwargs)
        marks = kwargs.pop('marks', [])
        for name in calculator_names:
            param = pytest.param((name, kwargs), marks=marks)
            calculator_inputs.append(param)
    if calculator_inputs:
        metafunc.parametrize('factory', calculator_inputs, indirect=True, ids=lambda input: input[0])