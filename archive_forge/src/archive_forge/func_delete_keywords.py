import os
import copy
from collections.abc import Iterable
from shutil import which
from typing import Dict, Optional
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator, EnvironmentError
def delete_keywords(self, kwargs):
    """removes list of keywords (delete) from kwargs"""
    for d in self.delete:
        kwargs.pop(d, None)