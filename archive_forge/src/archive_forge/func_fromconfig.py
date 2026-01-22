import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
@classmethod
def fromconfig(cls, config):
    return cls(config.executables['nwchem'])