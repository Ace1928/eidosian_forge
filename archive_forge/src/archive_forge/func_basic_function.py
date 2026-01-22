import os
from pytest import raises
from accelerate import PartialState, notebook_launcher
from accelerate.test_utils import require_bnb
from accelerate.utils import is_bnb_available
def basic_function():
    print(f'PartialState:\n{PartialState()}')