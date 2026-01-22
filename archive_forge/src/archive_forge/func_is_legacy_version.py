import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
def is_legacy_version(self):
    version = self.version()
    major_ver = int(version.split('.')[0])
    return major_ver < 9