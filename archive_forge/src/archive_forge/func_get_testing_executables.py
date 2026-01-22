import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
def get_testing_executables():
    paths = [Path.home() / '.config' / 'ase' / 'ase.conf']
    try:
        paths += [Path(x) for x in os.environ['ASE_CONFIG'].split(':')]
    except KeyError:
        pass
    conf = configparser.ConfigParser()
    conf['executables'] = {}
    effective_paths = conf.read(paths)
    return (effective_paths, conf['executables'])