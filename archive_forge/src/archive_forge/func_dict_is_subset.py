import pytest
from unittest import mock
import numpy as np
from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.calculators.vasp.create_input import _args_without_comment
from ase.calculators.vasp.create_input import _to_vasp_bool, _from_vasp_bool
from ase.build import bulk
def dict_is_subset(d1, d2):
    """True if all the key-value pairs in dict 1 are in dict 2"""
    return all((key in d2 and d1[key] == d2[key] for key in d1))